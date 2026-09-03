//! SGLang HiCache: the three-tier KV cache.
//!
//! HiCache extends RadixAttention into tiers — **L1 GPU HBM → L2 host memory →
//! L3 external storage** — moving pages between them transparently. For a
//! harness whose whole design is a large byte-stable prefix, L2 is the piece
//! that matters most: it lets a prefix far larger than VRAM stay warm.
//!
//! # What this module is and is not
//!
//! HiCache runs inside the SGLang server. QGI-2 cannot implement it and does
//! not try. What lives here is everything on *this* side of the boundary:
//!
//! - a typed model of the configuration, so it can be validated before a
//!   server is launched with it rather than after;
//! - launch-command generation, so the deployment and the harness cannot
//!   drift apart;
//! - tier-level metrics scraping, so an L2 that is thrashing is visible
//!   instead of showing up only as a slow turn;
//! - the page-alignment figure the assembler needs (see
//!   [`HiCacheConfig::page_size`] and `qgi2_assembler`).
//!
//! # Why the harness cares about `--page-size`
//!
//! HiCache stores KV in fixed-size pages. A prefix is reusable only up to its
//! last *complete* page: if the stable prefix is 1000 tokens and the page size
//! is 64, then 15 pages (960 tokens) are cacheable and the remaining 40 tokens
//! sit in a page shared with the start of the volatile tail — so that page is
//! recomputed every turn, however stable those 40 tokens are.
//!
//! Padding the stable prefix up to a page boundary converts that partial page
//! into a cached one. It is a small win per turn and a large one across a long
//! session, and it costs only the padding tokens themselves. The assembler does
//! this when given a [`PageAlignment`].

use crate::endpoint::Endpoint;
use crate::http::HttpClient;
use crate::metrics::prometheus_lines;
use qgi2_spec_types::Speculation;
use serde::{Deserialize, Serialize};
use std::fmt;

/// SGLang's default page size. Stated as a constant because the harness's
/// prefix alignment has to agree with the server's, and a mismatch is silent.
pub const DEFAULT_PAGE_SIZE: u32 = 64;

/// How L2 (host memory) is sized.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode", content = "value")]
pub enum L2Sizing {
    /// Host memory as a multiple of the GPU KV pool. `--hicache-ratio`.
    Ratio(f32),
    /// Absolute host memory in GB. `--hicache-size`, which overrides ratio.
    SizeGb(u32),
}

impl L2Sizing {
    pub fn flags(&self) -> Vec<String> {
        match self {
            Self::Ratio(r) => vec!["--hicache-ratio".into(), format!("{r}")],
            Self::SizeGb(g) => vec!["--hicache-size".into(), format!("{g}")],
        }
    }
}

/// The optional L3 backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "backend")]
pub enum L3Backend {
    /// Distributed KV pool with RDMA transfer. The backend that lets a prefix
    /// warmed on one host serve another.
    Mooncake {
        /// Mooncake master server address.
        master: String,
        /// Local hostname this node advertises.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        local_hostname: Option<String>,
        /// `rdma` or `tcp`.
        #[serde(default = "default_protocol")]
        protocol: String,
    },
    /// NVIDIA Inference Xfer Library.
    Nixl,
    /// A local directory. The simplest L3: survives a server restart, does not
    /// share between hosts.
    File { path: String },
    /// DeepSeek 3FS.
    Hf3fs,
    /// AIBrix distributed KV cache.
    AibrixKv,
}

fn default_protocol() -> String {
    "rdma".to_string()
}

impl L3Backend {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Mooncake { .. } => "mooncake",
            Self::Nixl => "nixl",
            Self::File { .. } => "file",
            Self::Hf3fs => "hf3fs",
            Self::AibrixKv => "aibrix",
        }
    }

    pub fn flags(&self) -> Vec<String> {
        vec!["--hicache-storage-backend".into(), self.name().into()]
    }

    /// Environment the backend needs alongside the flags.
    ///
    /// Mooncake is configured by environment rather than server flags, so a
    /// launch command alone is not enough to reproduce a working node.
    pub fn env(&self) -> Vec<(String, String)> {
        match self {
            Self::Mooncake {
                master,
                local_hostname,
                protocol,
            } => {
                let mut v = vec![
                    ("MOONCAKE_MASTER".to_string(), master.clone()),
                    ("MOONCAKE_PROTOCOL".to_string(), protocol.clone()),
                ];
                if let Some(h) = local_hostname {
                    v.push(("MOONCAKE_LOCAL_HOSTNAME".to_string(), h.clone()));
                }
                v
            }
            Self::File { path } => {
                vec![("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR".to_string(), path.clone())]
            }
            _ => Vec::new(),
        }
    }

    /// Whether this backend shares cache between hosts.
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Mooncake { .. } | Self::Hf3fs | Self::AibrixKv)
    }
}

/// How pages move between GPU and host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IoBackend {
    /// Kernel-based copy. The default recommendation.
    Kernel,
    /// Direct I/O.
    Direct,
}

impl IoBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Kernel => "kernel",
            Self::Direct => "direct",
        }
    }
}

/// When GPU pages are written down to host memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WritePolicy {
    /// Every page is written to L2 as it is produced. Highest L2 hit rate,
    /// highest write bandwidth.
    WriteThrough,
    /// Write only pages that have been reused. Cheaper, and the right default
    /// for a mixed workload.
    WriteThroughSelective,
    /// Write on eviction only. Lowest bandwidth, but a page evicted without
    /// being written is lost.
    WriteBack,
}

impl WritePolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WriteThrough => "write_through",
            Self::WriteThroughSelective => "write_through_selective",
            Self::WriteBack => "write_back",
        }
    }
}

/// KV memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemLayout {
    /// Zero-copy, recommended with the kernel I/O backend.
    PageFirst,
    /// Zero-copy and compatible with FlashAttention-3.
    PageFirstDirect,
    /// The pre-HiCache layout.
    LayerFirst,
}

impl MemLayout {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PageFirst => "page_first",
            Self::PageFirstDirect => "page_first_direct",
            Self::LayerFirst => "layer_first",
        }
    }
}

/// A full HiCache configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HiCacheConfig {
    pub enabled: bool,
    /// KV page size in tokens. The assembler aligns the stable prefix to this.
    pub page_size: u32,
    pub l2: L2Sizing,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub l3: Option<L3Backend>,
    pub io_backend: IoBackend,
    pub write_policy: WritePolicy,
    pub mem_layout: MemLayout,
    /// Whether the attention backend is FlashAttention-3, which constrains the
    /// usable memory layout.
    #[serde(default)]
    pub fa3: bool,
    /// Recurrent-state snapshot chunk for hybrid (GDN/Mamba) models, in tokens.
    /// Set this when the served model is a hybrid so the prefix is padded to a
    /// boundary the recurrence can actually resume from. `None` for
    /// pure-attention models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_chunk: Option<u32>,
    /// The recurrent-state pool for hybrid models. See [`GdnStatePool`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gdn: Option<GdnStatePool>,
}

impl Default for HiCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            page_size: DEFAULT_PAGE_SIZE,
            // A 2x host pool is the documented starting point: enough to hold
            // several prefixes without competing with the model for host RAM.
            l2: L2Sizing::Ratio(2.0),
            l3: None,
            io_backend: IoBackend::Kernel,
            // Selective, not write_through: QGI-2's prefix is reused constantly
            // and will be promoted immediately, while one-off volatile tails
            // never earn a write. Write_through would spend bandwidth on the
            // tail for no hit-rate gain.
            write_policy: WritePolicy::WriteThroughSelective,
            mem_layout: MemLayout::PageFirst,
            fa3: false,
            snapshot_chunk: None,
            gdn: None,
        }
    }
}

impl HiCacheConfig {
    /// L2 only: host-memory offload, no external storage.
    ///
    /// The right first step. It is most of the benefit on a single host and
    /// needs no extra infrastructure.
    pub fn l2_only() -> Self {
        Self::default()
    }

    /// L2 plus a local-file L3, so the cache survives a server restart.
    pub fn with_file_l3(path: impl Into<String>) -> Self {
        Self {
            l3: Some(L3Backend::File { path: path.into() }),
            ..Self::default()
        }
    }

    /// L2 plus a shared Mooncake L3, so a prefix warmed on one host serves the
    /// whole fleet.
    pub fn with_mooncake_l3(master: impl Into<String>, local_hostname: Option<String>) -> Self {
        Self {
            l3: Some(L3Backend::Mooncake {
                master: master.into(),
                local_hostname,
                protocol: default_protocol(),
            }),
            // A shared pool is only worth its bandwidth if pages actually reach
            // it, and selective writes withhold the first-use pages that another
            // host would most benefit from.
            write_policy: WritePolicy::WriteThrough,
            ..Self::default()
        }
    }

    /// The tiers this configuration actually enables.
    pub fn tiers(&self) -> Vec<Tier> {
        if !self.enabled {
            return vec![Tier::L1];
        }
        let mut t = vec![Tier::L1, Tier::L2];
        if self.l3.is_some() {
            t.push(Tier::L3);
        }
        t
    }

    /// Launch flags, in a stable order.
    pub fn launch_flags(&self) -> Vec<String> {
        if !self.enabled {
            return Vec::new();
        }
        let mut f = vec![
            "--page-size".into(),
            self.page_size.to_string(),
            "--enable-hierarchical-cache".into(),
        ];
        f.extend(self.l2.flags());
        f.push("--hicache-io-backend".into());
        f.push(self.io_backend.as_str().into());
        f.push("--hicache-write-policy".into());
        f.push(self.write_policy.as_str().into());
        f.push("--hicache-mem-layout".into());
        f.push(self.mem_layout.as_str().into());
        if let Some(l3) = &self.l3 {
            f.extend(l3.flags());
        }
        if let Some(g) = &self.gdn {
            f.extend(g.launch_flags());
        }
        f
    }

    /// Environment the configuration needs.
    pub fn env(&self) -> Vec<(String, String)> {
        self.l3.as_ref().map(|l| l.env()).unwrap_or_default()
    }

    /// Problems that would make this configuration behave unlike its author
    /// expects.
    ///
    /// Returned rather than logged: `qgi2 doctor` shows them, and a launch
    /// command should not be generated from a configuration that will not do
    /// what it says.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.enabled {
            return out;
        }

        if self.page_size == 0 || !self.page_size.is_power_of_two() {
            out.push(format!(
                "page_size {} should be a power of two; SGLang's default is {DEFAULT_PAGE_SIZE}",
                self.page_size
            ));
        }

        if let L2Sizing::Ratio(r) = self.l2
            && r <= 1.0
        {
            out.push(format!(
                "hicache-ratio {r} gives L2 no more room than the GPU pool, so nothing can be \
                 held that would not already fit in L1 — the offload will not help"
            ));
        }

        if self.fa3 && self.mem_layout == MemLayout::PageFirst {
            out.push(
                "mem_layout page_first is not compatible with FlashAttention-3; \
                 use page_first_direct"
                    .into(),
            );
        }

        if self.mem_layout == MemLayout::LayerFirst {
            out.push(
                "mem_layout layer_first forgoes the zero-copy path; page_first \
                 (or page_first_direct under fa3) is what makes L2 transfers cheap"
                    .into(),
            );
        }

        if self.write_policy == WritePolicy::WriteBack && self.l3.is_some() {
            out.push(
                "write_back with an L3 backend can lose pages that are evicted before they are \
                 written down, which is the opposite of what an L3 is for"
                    .into(),
            );
        }

        if let Some(L3Backend::Mooncake { master, .. }) = &self.l3
            && master.trim().is_empty()
        {
            out.push("the mooncake L3 backend needs a master address".into());
        }

        if let Some(g) = &self.gdn {
            out.extend(g.problems());
        }

        if let Some(chunk) = self.snapshot_chunk
            && self.page_size > 0
            && chunk % self.page_size != 0
        {
            // Engines that snapshot recurrent state require the KV page and the
            // state boundary to coincide (FreeToken asserts CHUNK_SIZE %
            // page_size == 0); a misaligned pair fails at launch.
            out.push(format!(
                "snapshot_chunk {chunk} is not a multiple of page_size {}; hybrid engines \
                 require the recurrent-state boundary to land on a page boundary",
                self.page_size
            ));
        }

        out
    }

    /// A paste-able launch command.
    pub fn launch_command(&self, model_path: &str, port: u16, extra: &[String]) -> String {
        let mut s = String::new();
        for (k, v) in self.env() {
            s.push_str(&format!("{k}={v} \\\n  "));
        }
        s.push_str(&format!(
            "python -m sglang.launch_server \\\n  --model-path {model_path} \\\n  --host 0.0.0.0 --port {port}"
        ));
        for line in group_flags(&self.launch_flags()) {
            s.push_str(" \\\n  ");
            s.push_str(&line);
        }
        for line in group_flags(extra) {
            s.push_str(" \\\n  ");
            s.push_str(&line);
        }
        s
    }

    /// The alignment the assembler should pad the stable prefix to.
    pub fn page_alignment(&self) -> Option<PageAlignment> {
        self.enabled.then_some(PageAlignment {
            page_size: self.page_size,
            snapshot_chunk: self.snapshot_chunk,
        })
    }
}

/// Group a flat flag list into one line per flag, pairing each with its value.
///
/// `chunks(2)` cannot be used: `--enable-hierarchical-cache` takes no value, so
/// a fixed pairing puts it with the next flag's name and splits every later
/// flag from its own value. The result is still valid shell, which is what
/// makes the mistake easy to miss by eye.
fn group_flags(flags: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < flags.len() {
        let name = &flags[i];
        let takes_value = flags
            .get(i + 1)
            .is_some_and(|next| !next.starts_with("--"));
        if takes_value {
            out.push(format!("{name} {}", flags[i + 1]));
            i += 2;
        } else {
            out.push(name.clone());
            i += 1;
        }
    }
    out
}

/// The recurrent-state pool a hybrid (GDN/Mamba) model needs, and the VRAM it
/// takes from the KV pool.
///
/// Borrowed from the RTX PRO 6000 DSpark recipe's calculator
/// (SamSammane/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark, `start.sh`), which is the
/// SGLang cookbook's mamba-ratio calculator worked for 96 GB.
///
/// # Why this is its own resource
///
/// A pure-attention model has one cache: KV, paged, evictable. A hybrid model
/// has a second one that behaves nothing like it. Each running request pins a
/// fixed number of recurrent-state **slots** for its lifetime -- no paging, no
/// eviction -- and a slot is large: **78.4 MB** on the 27B (48 GDN layers x 48
/// heads x 128 x 128 bf16, plus conv state). Speculation multiplies it: each
/// draft token in flight needs its own slot, so a DSpark block of 7 wants 8
/// draft slots on top of the 4 the lazy radix strategy keeps.
///
/// So the pool caps **concurrency** independently of KV. On 96 GB, KV stops
/// being the bound before the state pool does; the recipe pins the pool
/// explicitly (`--max-mamba-cache-size`) rather than letting the ratio flag
/// guess. QGI-2 models it for the same reason it models page alignment: a
/// number the engine will enforce is a number the harness should compute
/// before launch, not discover at it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GdnStatePool {
    /// Bytes per recurrent-state slot. 78.4 MB for Qwen3.8-27B at bf16.
    pub slot_bytes: u64,
    /// Slots the radix strategy keeps per request beyond the drafts. 4 for
    /// `extra_buffer_lazy`.
    pub lazy_slots: u32,
    /// Concurrent requests the pool must serve.
    pub max_concurrent: u32,
    /// Speculation in force, which sets the draft slots per request.
    pub speculation: Speculation,
    /// VRAM budget, for the fit check. `None` skips it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<VramBudget>,
}

/// The VRAM arithmetic from the recipe: what is left for KV once weights, the
/// state pool, and the runtime have taken theirs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VramBudget {
    pub total_gb: f64,
    /// `--mem-fraction-static`. 0.90 in the recipe.
    pub mem_fraction_static: f64,
    /// Base weights plus any drafter. 24.5 GB in the recipe (21.9 + 2.7).
    pub weights_gb: f64,
    /// CUDA graphs, FlashInfer workspace, mm pools. 3.5 GB in the recipe.
    pub runtime_gb: f64,
}

impl GdnStatePool {
    /// The 27B's numbers from the recipe, for a given concurrency and
    /// speculation.
    pub fn qwen38_27b(max_concurrent: u32, speculation: Speculation) -> Self {
        Self {
            slot_bytes: 78_400_000,
            lazy_slots: 4,
            max_concurrent,
            speculation,
            budget: None,
        }
    }

    pub fn with_budget(mut self, budget: VramBudget) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Draft slots a speculator holds in flight: the verify width, which is
    /// the block size plus the bonus token. Zero when not speculating.
    pub fn draft_slots(&self) -> u32 {
        match self.speculation {
            Speculation::Off => 0,
            s => u32::from(s.lookahead()) + 1,
        }
    }

    /// `S + D`: lazy buffer plus drafts.
    pub fn slots_per_request(&self) -> u32 {
        self.lazy_slots + self.draft_slots()
    }

    /// `--max-mamba-cache-size`.
    pub fn total_slots(&self) -> u32 {
        self.max_concurrent * self.slots_per_request()
    }

    pub fn pool_bytes(&self) -> u64 {
        u64::from(self.total_slots()) * self.slot_bytes
    }

    pub fn pool_gb(&self) -> f64 {
        self.pool_bytes() as f64 / 1e9
    }

    /// GB left for the KV pool under the budget, if one is set.
    pub fn kv_pool_gb(&self) -> Option<f64> {
        let b = self.budget?;
        Some(b.total_gb * b.mem_fraction_static - b.weights_gb - self.pool_gb() - b.runtime_gb)
    }

    /// The flags that pin the pool, from the recipe.
    pub fn launch_flags(&self) -> Vec<String> {
        vec![
            "--mamba-ssm-dtype".into(),
            "bfloat16".into(),
            "--mamba-radix-cache-strategy".into(),
            "extra_buffer_lazy".into(),
            "--max-mamba-cache-size".into(),
            self.total_slots().to_string(),
            "--max-running-requests".into(),
            self.max_concurrent.to_string(),
        ]
    }

    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.max_concurrent == 0 {
            out.push("gdn pool: max_concurrent is 0; nothing could run".into());
        }
        if let Some(kv) = self.kv_pool_gb() {
            if kv <= 0.0 {
                out.push(format!(
                    "gdn pool: {} concurrent x {} slots x {:.1} MB = {:.1} GB of state leaves no \
                     VRAM for KV under the budget; lower concurrency or the draft block",
                    self.max_concurrent,
                    self.slots_per_request(),
                    self.slot_bytes as f64 / 1e6,
                    self.pool_gb()
                ));
            } else if kv < 8.0 {
                // ~1.5M tokens at 32.8 KB/token is the recipe's KV pool; under
                // 8 GB (~250K tokens) a single long-context request cannot
                // reach the model's native window.
                out.push(format!(
                    "gdn pool: only {kv:.1} GB left for KV after a {:.1} GB state pool; a single \
                     request cannot use the model's context window",
                    self.pool_gb()
                ));
            }
        }
        out
    }
}

/// A cache tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// GPU HBM.
    L1,
    /// Host memory.
    L2,
    /// External storage.
    L3,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::L1 => "L1 (GPU)",
            Self::L2 => "L2 (host)",
            Self::L3 => "L3 (storage)",
        })
    }
}

/// How the stable prefix is padded to a cache boundary.
///
/// Pages are counted in **tokens**, and QGI-2 does not tokenize — it works in
/// bytes. So the padding is computed from an estimate, and the estimate being
/// wrong costs at most one unit rather than breaking anything. The
/// authoritative check is the `cached_tokens` the engine reports back; if the
/// alignment is not paying off, that number says so.
///
/// # Two granularities, not one
///
/// A pure-attention model's reusable prefix is page-granular. A **hybrid**
/// model — GDN/Mamba linear layers beside attention, which is what the spec's
/// own planner (Flash-Next: "GDN + QSA hybrid") is — is not. Its recurrent
/// state cannot be resumed mid-stream; an engine checkpoints it only at
/// `snapshot_chunk`-aligned boundaries and truncates the reusable prefix to the
/// deepest live snapshot (FreeToken's hybrid radix cache, mirroring SGLang's
/// `MambaRadixCache`). Padding to a page boundary that is not also a snapshot
/// boundary still forces the recurrence to replay from the previous chunk.
///
/// So the effective unit is `lcm(page_size, snapshot_chunk)`. With the common
/// defaults (64, 64) that is 64 and nothing changes; DeepSeek-V4's 128-token
/// window pages make it 128; a 16-token page under a 64-token chunk makes it
/// 64, not 16.
///
/// # A failure alignment cannot prevent, and the harness makes reachable
///
/// syv-ai/qwen38-27b-rtx3090, gotcha 37: on vLLM with prefix caching, a
/// captured verify step and a draft count `k`, a request that **hits** the
/// prefix cache and whose *total* prompt length is `≡ 117 + k (mod 128)`
/// collapses -- DFlash2 to 1.97 tok/step with degenerate repetition, MTP to an
/// empty answer with `finish_reason: stop`. Every other residue is clean.
///
/// Alignment pads the *stable prefix*; the total length also includes the
/// volatile tail and so varies per turn, landing on the bad residue roughly
/// one turn in 128. QGI-2 engineers cache hits deliberately, which is exactly
/// the precondition. Nothing here can prevent it. What the harness can do is
/// not hide it: a turn whose acceptance drops to ~2.0 with a near-perfect cache
/// hit is this bug, not a cache problem, and the fix is the engine's. Do not
/// sample residues to check -- five samples miss one bad residue 96% of the
/// time; walk all 128 by padding one token at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageAlignment {
    pub page_size: u32,
    /// Recurrent-state snapshot granularity for hybrid models, in tokens.
    /// `None` for pure-attention models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_chunk: Option<u32>,
}

impl PageAlignment {
    pub const fn pages(page_size: u32) -> Self {
        Self {
            page_size,
            snapshot_chunk: None,
        }
    }

    /// For a hybrid model whose engine snapshots recurrent state every
    /// `snapshot_chunk` tokens.
    pub const fn hybrid(page_size: u32, snapshot_chunk: u32) -> Self {
        Self {
            page_size,
            snapshot_chunk: Some(snapshot_chunk),
        }
    }

    /// The boundary the prefix is actually padded to: the least common multiple
    /// of the page and, when present, the snapshot chunk.
    pub fn granularity(&self) -> u32 {
        match self.snapshot_chunk {
            Some(c) if c > 0 && self.page_size > 0 => lcm(self.page_size, c),
            _ => self.page_size,
        }
    }
    /// Characters per token, for converting a byte length into a token
    /// estimate. ~3.6 is typical of English prose and code under a BPE
    /// tokenizer; it is deliberately a little low, because over-padding wastes
    /// a few tokens while under-padding wastes the whole page.
    pub const CHARS_PER_TOKEN: f32 = 3.6;

    pub fn estimated_tokens(&self, byte_len: usize) -> u32 {
        (byte_len as f32 / Self::CHARS_PER_TOKEN).ceil() as u32
    }

    /// Tokens of padding that would carry `byte_len` up to a page boundary.
    ///
    /// Returns 0 when the prefix already lands on one, and never pads a whole
    /// page for nothing.
    pub fn padding_tokens(&self, byte_len: usize) -> u32 {
        let unit = self.granularity();
        if unit == 0 {
            return 0;
        }
        let tokens = self.estimated_tokens(byte_len);
        let remainder = tokens % unit;
        if remainder == 0 {
            0
        } else {
            unit - remainder
        }
    }

    /// Padding text that carries a prefix of `byte_len` onto a page boundary.
    ///
    /// The byte count is derived from the *target* token count rather than from
    /// the padding token count, because [`Self::estimated_tokens`] rounds up:
    /// generating "enough" bytes and stopping at the first length that covers
    /// them overshoots, and the ceiling then pushes the total one token past
    /// the boundary — landing on exactly the partial page the padding exists to
    /// avoid.
    ///
    /// A repeated comment line rather than whitespace: tokenizers collapse runs
    /// of spaces unpredictably, so whitespace padding would not reliably occupy
    /// the tokens it appears to. The text is inert to the model and identical
    /// for a given length, which keeps the prefix byte-stable.
    pub fn padding_text(&self, byte_len: usize) -> String {
        let pad_bytes = self.padding_bytes(byte_len);
        if pad_bytes == 0 {
            return String::new();
        }
        const UNIT: &str = "\n# pad";
        let mut s = String::with_capacity(pad_bytes + UNIT.len());
        while s.len() < pad_bytes {
            s.push_str(UNIT);
        }
        // UNIT is ASCII, so truncating to a byte length cannot split a
        // character.
        s.truncate(pad_bytes);
        s
    }

    /// Bytes of padding needed to land `byte_len` on a page boundary.
    ///
    /// Chosen so that `estimated_tokens(byte_len + padding_bytes)` is exactly
    /// the next page multiple: the largest byte count that still rounds up to
    /// the target token count.
    pub fn padding_bytes(&self, byte_len: usize) -> usize {
        let pad_tokens = self.padding_tokens(byte_len);
        if pad_tokens == 0 {
            return 0;
        }
        let target_tokens = self.estimated_tokens(byte_len) + pad_tokens;
        let max_bytes = (target_tokens as f32 * Self::CHARS_PER_TOKEN).floor() as usize;
        max_bytes.saturating_sub(byte_len)
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: u32, b: u32) -> u32 {
    a / gcd(a, b) * b
}

/// Per-tier hit rates, as SGLang reports them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct HiCacheStats {
    /// RadixAttention prefix hit rate (L1).
    pub l1_hit_rate: Option<f64>,
    pub l2_hit_rate: Option<f64>,
    pub l3_hit_rate: Option<f64>,
    /// Host-memory pool utilisation, if reported.
    pub host_pool_utilization: Option<f64>,
}

impl HiCacheStats {
    pub fn any(&self) -> bool {
        self.l1_hit_rate.is_some() || self.l2_hit_rate.is_some() || self.l3_hit_rate.is_some()
    }

    /// Findings worth acting on.
    pub fn findings(&self, config: &HiCacheConfig) -> Vec<String> {
        let mut out = Vec::new();

        if config.enabled && self.l2_hit_rate.is_none() && self.any() {
            out.push(
                "L2 is configured but reports no hit rate — the server may not have \
                 hierarchical caching enabled even though the harness thinks it does"
                    .into(),
            );
        }
        if let Some(l2) = self.l2_hit_rate
            && l2 < 0.05
            && self.l1_hit_rate.is_some_and(|l1| l1 > 0.5)
        {
            // A high L1 with a dead L2 usually means the working set already
            // fits in GPU memory, so the offload is pure overhead.
            out.push(format!(
                "L2 hit rate {:.1}% is negligible while L1 is healthy — the working set fits \
                 in GPU memory, so hicache-ratio could come down",
                l2 * 100.0
            ));
        }
        if let Some(u) = self.host_pool_utilization
            && u > 0.95
        {
            out.push(format!(
                "host pool is {:.0}% full; raise hicache-ratio or add an L3 backend before \
                 pages start being evicted before reuse",
                u * 100.0
            ));
        }
        if config.l3.is_some() && self.l3_hit_rate == Some(0.0) {
            out.push(
                "L3 is configured but has never hit — check the backend is reachable, and that \
                 write_policy is not withholding pages from it"
                    .into(),
            );
        }
        out
    }
}

/// Scrape per-tier statistics from an SGLang process.
///
/// Metric names have moved between SGLang releases, so several spellings are
/// accepted per tier and an unrecognised exposition yields "no data" rather
/// than an error — a HiCache reading is diagnostic, and a turn should not fail
/// because it could not be taken.
pub async fn scrape_hicache(http: &HttpClient, endpoint: &Endpoint) -> HiCacheStats {
    let url = format!("{}/metrics", endpoint.root());
    let Ok(body) = http.get_text(&url, endpoint.api_key.as_deref()).await else {
        return HiCacheStats::default();
    };
    parse_hicache(&body)
}

/// Parse tier statistics out of Prometheus text.
pub fn parse_hicache(body: &str) -> HiCacheStats {
    let mut s = HiCacheStats::default();
    for (name, value) in prometheus_lines(body) {
        match name {
            "sglang:cache_hit_rate" | "sglang:cached_tokens_rate" => {
                s.l1_hit_rate = Some(normalize_rate(value))
            }
            "sglang:hicache_l2_hit_rate" | "sglang:host_cache_hit_rate" => {
                s.l2_hit_rate = Some(normalize_rate(value))
            }
            "sglang:hicache_l3_hit_rate" | "sglang:storage_cache_hit_rate" => {
                s.l3_hit_rate = Some(normalize_rate(value))
            }
            "sglang:host_kv_cache_usage" | "sglang:hicache_host_mem_usage" => {
                s.host_pool_utilization = Some(normalize_rate(value))
            }
            _ => {}
        }
    }
    s
}

/// Some builds report a percentage and some a fraction. A value above 1 is a
/// percentage; taking it at face value would make every hit rate look like a
/// catastrophic breach or an impossible success.
fn normalize_rate(v: f64) -> f64 {
    if v > 1.0 { (v / 100.0).min(1.0) } else { v.max(0.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_only_is_the_default_and_enables_two_tiers() {
        let c = HiCacheConfig::l2_only();
        assert_eq!(c.tiers(), vec![Tier::L1, Tier::L2]);
        assert!(c.l3.is_none());
        assert!(c.problems().is_empty(), "{:?}", c.problems());
    }

    #[test]
    fn a_file_l3_adds_the_third_tier_and_its_directory() {
        let c = HiCacheConfig::with_file_l3("/var/cache/qgi2");
        assert_eq!(c.tiers(), vec![Tier::L1, Tier::L2, Tier::L3]);
        assert!(c.launch_flags().contains(&"file".to_string()));
        assert_eq!(c.env()[0].1, "/var/cache/qgi2");
        assert!(!c.l3.as_ref().unwrap().is_shared());
    }

    #[test]
    fn a_mooncake_l3_is_shared_and_writes_through() {
        // Selective writes withhold exactly the first-use pages another host
        // would benefit from, so a shared pool wants write_through.
        let c = HiCacheConfig::with_mooncake_l3("10.0.0.1:50051", Some("gpu-1".into()));
        assert!(c.l3.as_ref().unwrap().is_shared());
        assert_eq!(c.write_policy, WritePolicy::WriteThrough);
        let env: Vec<_> = c.env().into_iter().map(|(k, _)| k).collect();
        assert!(env.contains(&"MOONCAKE_MASTER".to_string()));
        assert!(env.contains(&"MOONCAKE_LOCAL_HOSTNAME".to_string()));
    }

    #[test]
    fn launch_flags_include_every_tier_setting() {
        let flags = HiCacheConfig::with_file_l3("/tmp/kv").launch_flags().join(" ");
        for expected in [
            "--page-size",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "--hicache-io-backend",
            "--hicache-write-policy",
            "--hicache-mem-layout",
            "--hicache-storage-backend",
        ] {
            assert!(flags.contains(expected), "missing {expected} in {flags}");
        }
    }

    #[test]
    fn disabling_hicache_emits_no_flags() {
        let c = HiCacheConfig {
            enabled: false,
            ..HiCacheConfig::default()
        };
        assert!(c.launch_flags().is_empty());
        assert_eq!(c.tiers(), vec![Tier::L1]);
        assert!(c.page_alignment().is_none());
    }

    #[test]
    fn a_useless_ratio_is_flagged() {
        let c = HiCacheConfig {
            l2: L2Sizing::Ratio(1.0),
            ..HiCacheConfig::default()
        };
        assert!(c.problems()[0].contains("will not help"), "{:?}", c.problems());
    }

    #[test]
    fn fa3_requires_the_direct_layout() {
        let c = HiCacheConfig {
            fa3: true,
            mem_layout: MemLayout::PageFirst,
            ..HiCacheConfig::default()
        };
        assert!(c.problems().iter().any(|p| p.contains("page_first_direct")));

        let ok = HiCacheConfig {
            fa3: true,
            mem_layout: MemLayout::PageFirstDirect,
            ..HiCacheConfig::default()
        };
        assert!(ok.problems().is_empty(), "{:?}", ok.problems());
    }

    #[test]
    fn write_back_with_an_l3_is_flagged_as_self_defeating() {
        let c = HiCacheConfig {
            write_policy: WritePolicy::WriteBack,
            ..HiCacheConfig::with_file_l3("/tmp/kv")
        };
        assert!(c.problems().iter().any(|p| p.contains("opposite")));
    }

    #[test]
    fn a_mooncake_backend_without_a_master_is_flagged() {
        let c = HiCacheConfig {
            l3: Some(L3Backend::Mooncake {
                master: "  ".into(),
                local_hostname: None,
                protocol: "rdma".into(),
            }),
            ..HiCacheConfig::default()
        };
        assert!(c.problems().iter().any(|p| p.contains("master address")));
    }

    #[test]
    fn bare_flags_do_not_desynchronise_the_command() {
        // The bug this pins: pairing flags two at a time puts the valueless
        // --enable-hierarchical-cache with the next flag's *name*, splitting
        // every later flag from its value. It still parses as shell, which is
        // what makes it easy to miss.
        let lines = group_flags(&[
            "--page-size".into(),
            "64".into(),
            "--enable-hierarchical-cache".into(),
            "--hicache-ratio".into(),
            "2".into(),
        ]);
        assert_eq!(
            lines,
            vec![
                "--page-size 64".to_string(),
                "--enable-hierarchical-cache".to_string(),
                "--hicache-ratio 2".to_string(),
            ]
        );
    }

    #[test]
    fn every_flag_line_keeps_its_value() {
        for line in group_flags(&HiCacheConfig::with_file_l3("/tmp/kv").launch_flags()) {
            assert!(line.starts_with("--"), "orphaned value: {line:?}");
            let parts: Vec<_> = line.split(' ').collect();
            assert!(parts.len() <= 2, "flag line has stray tokens: {line:?}");
            if parts.len() == 2 {
                assert!(!parts[1].starts_with("--"), "flag paired with a flag: {line:?}");
            }
        }
    }

    #[test]
    fn the_launch_command_is_paste_able() {
        let cmd = HiCacheConfig::with_mooncake_l3("m:50051", None).launch_command(
            "/models/worker",
            30000,
            &["--speculative-algorithm".into(), "EAGLE3".into()],
        );
        assert!(cmd.starts_with("MOONCAKE_MASTER=m:50051"));
        assert!(cmd.contains("python -m sglang.launch_server"));
        assert!(cmd.contains("--model-path /models/worker"));
        assert!(cmd.contains("--port 30000"));
        assert!(cmd.contains("EAGLE3"));
    }

    // --- page alignment ---

    #[test]
    fn a_prefix_already_on_a_boundary_is_not_padded() {
        let a = PageAlignment::pages(64);
        // 64 tokens' worth of bytes.
        let bytes = (64.0 * PageAlignment::CHARS_PER_TOKEN) as usize;
        assert_eq!(a.padding_tokens(bytes), 0);
        assert!(a.padding_text(bytes).is_empty());
    }

    #[test]
    fn a_partial_page_is_padded_up_to_the_boundary() {
        let a = PageAlignment::pages(64);
        // ~100 tokens: 36 short of the second page.
        let bytes = (100.0 * PageAlignment::CHARS_PER_TOKEN) as usize;
        let pad = a.padding_tokens(bytes);
        assert!(pad > 0 && pad < 64, "pad was {pad}");
        assert_eq!(a.estimated_tokens(bytes) + pad, 128);
    }

    #[test]
    fn padding_never_exceeds_one_page() {
        let a = PageAlignment::pages(64);
        for bytes in [0, 1, 100, 999, 5000, 100_000] {
            assert!(a.padding_tokens(bytes) < 64, "bytes={bytes}");
        }
    }

    #[test]
    fn padding_is_deterministic_for_a_given_length() {
        let a = PageAlignment::pages(64);
        assert_eq!(a.padding_text(1000), a.padding_text(1000));
    }

    #[test]
    fn padding_is_not_whitespace() {
        // Tokenizers collapse runs of spaces unpredictably, so whitespace would
        // not reliably occupy the tokens it appears to.
        let a = PageAlignment::pages(64);
        let text = a.padding_text(1000);
        assert!(!text.trim().is_empty());
        assert!(text.contains("pad"));
    }

    #[test]
    fn a_zero_page_size_pads_nothing_rather_than_dividing_by_zero() {
        let a = PageAlignment::pages(0);
        assert_eq!(a.padding_tokens(1234), 0);
    }

    // --- metrics ---

    #[test]
    fn tier_hit_rates_parse() {
        let body = "\
sglang:cache_hit_rate{model=\"w\"} 0.91
sglang:hicache_l2_hit_rate{model=\"w\"} 0.42
sglang:hicache_l3_hit_rate{model=\"w\"} 0.08
sglang:host_kv_cache_usage{model=\"w\"} 0.6
";
        let s = parse_hicache(body);
        assert_eq!(s.l1_hit_rate, Some(0.91));
        assert_eq!(s.l2_hit_rate, Some(0.42));
        assert_eq!(s.l3_hit_rate, Some(0.08));
        assert_eq!(s.host_pool_utilization, Some(0.6));
    }

    #[test]
    fn a_percentage_is_normalized_to_a_fraction() {
        // Taking 91.0 at face value would read as an impossible 9100%.
        let s = parse_hicache("sglang:cache_hit_rate 91.0\n");
        assert_eq!(s.l1_hit_rate, Some(0.91));
    }

    #[test]
    fn unrecognised_output_yields_no_data_rather_than_zeros() {
        let s = parse_hicache("some_other_metric 1.0\n");
        assert!(!s.any());
        assert_eq!(s.l2_hit_rate, None);
    }

    #[test]
    fn a_dead_l2_beside_a_healthy_l1_is_reported_as_wasted_ratio() {
        let s = HiCacheStats {
            l1_hit_rate: Some(0.9),
            l2_hit_rate: Some(0.01),
            ..HiCacheStats::default()
        };
        let f = s.findings(&HiCacheConfig::l2_only());
        assert!(f.iter().any(|m| m.contains("hicache-ratio could come down")), "{f:?}");
    }

    #[test]
    fn a_full_host_pool_is_reported_before_it_starts_thrashing() {
        let s = HiCacheStats {
            l1_hit_rate: Some(0.9),
            l2_hit_rate: Some(0.4),
            host_pool_utilization: Some(0.99),
            ..HiCacheStats::default()
        };
        assert!(
            s.findings(&HiCacheConfig::l2_only())
                .iter()
                .any(|m| m.contains("host pool"))
        );
    }

    #[test]
    fn an_l3_that_never_hits_is_reported() {
        let s = HiCacheStats {
            l1_hit_rate: Some(0.9),
            l2_hit_rate: Some(0.4),
            l3_hit_rate: Some(0.0),
            ..HiCacheStats::default()
        };
        let f = s.findings(&HiCacheConfig::with_file_l3("/tmp/kv"));
        assert!(f.iter().any(|m| m.contains("never hit")), "{f:?}");
    }

    #[test]
    fn the_config_round_trips_through_serde() {
        let c = HiCacheConfig::with_mooncake_l3("m:1", Some("h".into()));
        let back: HiCacheConfig = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
    }
}

#[cfg(test)]
mod alignment_math_tests {
    use super::*;

    #[test]
    fn padding_lands_exactly_on_a_page_boundary() {
        // The bug this pins: generating "enough" padding bytes and stopping at
        // the first length that covers them overshoots, and estimated_tokens'
        // ceiling then pushes the total one token past the boundary — onto the
        // exact partial page the padding exists to avoid.
        let a = PageAlignment::pages(64);
        for prefix in (1..4000).step_by(7) {
            let padded = prefix + a.padding_bytes(prefix);
            let tokens = a.estimated_tokens(padded);
            assert_eq!(
                tokens % 64,
                0,
                "prefix {prefix} padded to {padded} bytes = {tokens} tokens"
            );
        }
    }

    #[test]
    fn the_generated_text_is_exactly_the_computed_length() {
        let a = PageAlignment::pages(64);
        for prefix in [100usize, 500, 1234, 9999] {
            assert_eq!(a.padding_text(prefix).len(), a.padding_bytes(prefix));
        }
    }

    #[test]
    fn padding_costs_less_than_one_page_of_bytes() {
        let a = PageAlignment::pages(64);
        let page_bytes = (64.0 * PageAlignment::CHARS_PER_TOKEN) as usize;
        for prefix in [1usize, 100, 1000, 50_000] {
            assert!(a.padding_bytes(prefix) <= page_bytes, "prefix={prefix}");
        }
    }

    #[test]
    fn an_already_aligned_prefix_gets_no_padding() {
        let a = PageAlignment::pages(64);
        let aligned = (64.0 * PageAlignment::CHARS_PER_TOKEN) as usize;
        assert_eq!(a.padding_bytes(aligned), 0);
        assert!(a.padding_text(aligned).is_empty());
    }
}


#[cfg(test)]
mod granularity_tests {
    use super::*;

    #[test]
    fn a_pure_attention_model_aligns_to_the_page() {
        assert_eq!(PageAlignment::pages(64).granularity(), 64);
    }

    #[test]
    fn matching_chunk_and_page_change_nothing() {
        // FreeToken's CHUNK_SIZE is 64, as is the default page: the common case
        // is unaffected by the generalisation.
        assert_eq!(PageAlignment::hybrid(64, 64).granularity(), 64);
    }

    #[test]
    fn a_larger_chunk_wins() {
        // A hybrid model's recurrence can only resume at a chunk boundary, so a
        // prefix padded merely to a page still replays from the last chunk.
        assert_eq!(PageAlignment::hybrid(64, 128).granularity(), 128);
        let a = PageAlignment::hybrid(64, 128);
        for prefix in (1..4000).step_by(11) {
            let padded = prefix + a.padding_bytes(prefix);
            assert_eq!(a.estimated_tokens(padded) % 128, 0, "prefix {prefix}");
        }
    }

    #[test]
    fn a_smaller_page_still_aligns_to_the_chunk() {
        // page 16 under a 64-token chunk: the unit is 64, not 16.
        assert_eq!(PageAlignment::hybrid(16, 64).granularity(), 64);
    }

    #[test]
    fn coprime_sizes_take_the_lcm() {
        assert_eq!(PageAlignment::hybrid(64, 96).granularity(), 192);
    }

    #[test]
    fn a_zero_chunk_falls_back_to_the_page() {
        assert_eq!(PageAlignment::hybrid(64, 0).granularity(), 64);
    }

    #[test]
    fn a_misaligned_chunk_is_flagged_before_launch() {
        // FreeToken asserts CHUNK_SIZE % page_size == 0 at startup; catching it
        // in config beats a crash on the launch line.
        let c = HiCacheConfig {
            page_size: 48,
            snapshot_chunk: Some(64),
            ..HiCacheConfig::default()
        };
        assert!(c.problems().iter().any(|p| p.contains("snapshot_chunk")), "{:?}", c.problems());
        let ok = HiCacheConfig {
            page_size: 64,
            snapshot_chunk: Some(128),
            ..HiCacheConfig::default()
        };
        assert!(!ok.problems().iter().any(|p| p.contains("snapshot_chunk")));
    }

    #[test]
    fn the_config_hands_the_chunk_to_the_alignment() {
        let c = HiCacheConfig {
            snapshot_chunk: Some(128),
            ..HiCacheConfig::default()
        };
        assert_eq!(c.page_alignment().unwrap().granularity(), 128);
    }
}


#[cfg(test)]
mod gdn_tests {
    use super::*;

    fn recipe_budget() -> VramBudget {
        VramBudget {
            total_gb: 96.0,
            mem_fraction_static: 0.90,
            weights_gb: 24.5,
            runtime_gb: 3.5,
        }
    }

    #[test]
    fn the_recipe_calculator_reproduces() {
        // start.sh: S=4 + D=8 = 12 slots/req; 8 concurrent -> 96 slots -> ~7.5 GB;
        // 86.4 - 24.5 - 7.5 - 3.5 = ~50 GB KV.
        let g = GdnStatePool::qwen38_27b(8, Speculation::DSpark { n: 7 })
            .with_budget(recipe_budget());
        assert_eq!(g.draft_slots(), 8);
        assert_eq!(g.slots_per_request(), 12);
        assert_eq!(g.total_slots(), 96);
        assert!((g.pool_gb() - 7.53).abs() < 0.05, "{}", g.pool_gb());
        let kv = g.kv_pool_gb().unwrap();
        assert!((kv - 50.9).abs() < 0.5, "{kv}");
        assert!(g.problems().is_empty(), "{:?}", g.problems());
    }

    #[test]
    fn no_speculation_means_no_draft_slots() {
        let g = GdnStatePool::qwen38_27b(8, Speculation::Off);
        assert_eq!(g.draft_slots(), 0);
        assert_eq!(g.slots_per_request(), 4);
    }

    #[test]
    fn a_longer_draft_block_costs_slots_not_tokens() {
        // rtx3090 gotcha 23: the verify block costs KV pool per request slot.
        let short = GdnStatePool::qwen38_27b(8, Speculation::DFlash2 { n: 7 });
        let long = GdnStatePool::qwen38_27b(8, Speculation::DFlash2 { n: 15 });
        assert_eq!(short.slots_per_request(), 12);
        assert_eq!(long.slots_per_request(), 20);
        assert!(long.pool_gb() > short.pool_gb() * 1.6);
    }

    #[test]
    fn the_pool_caps_concurrency_before_kv_does() {
        // Push concurrency until the state pool eats the KV budget: the harness
        // should say so before launch rather than the engine refusing to start.
        let g = GdnStatePool::qwen38_27b(64, Speculation::DSpark { n: 7 })
            .with_budget(recipe_budget());
        let p = g.problems();
        assert!(!p.is_empty(), "64 concurrent x 12 slots is {:.1} GB", g.pool_gb());
        assert!(p[0].contains("leaves no VRAM for KV") || p[0].contains("only"), "{p:?}");
    }

    #[test]
    fn launch_flags_pin_the_pool() {
        let g = GdnStatePool::qwen38_27b(8, Speculation::DSpark { n: 7 });
        let f = g.launch_flags().join(" ");
        assert!(f.contains("--max-mamba-cache-size 96"), "{f}");
        assert!(f.contains("--mamba-radix-cache-strategy extra_buffer_lazy"), "{f}");
        assert!(f.contains("--max-running-requests 8"), "{f}");
    }

    #[test]
    fn the_hicache_config_carries_the_pool_into_its_flags_and_problems() {
        let c = HiCacheConfig {
            gdn: Some(
                GdnStatePool::qwen38_27b(8, Speculation::DSpark { n: 7 })
                    .with_budget(recipe_budget()),
            ),
            ..HiCacheConfig::default()
        };
        assert!(c.launch_flags().join(" ").contains("--max-mamba-cache-size 96"));
        assert!(c.problems().is_empty(), "{:?}", c.problems());

        let too_many = HiCacheConfig {
            gdn: Some(
                GdnStatePool::qwen38_27b(64, Speculation::DSpark { n: 7 })
                    .with_budget(recipe_budget()),
            ),
            ..HiCacheConfig::default()
        };
        assert!(!too_many.problems().is_empty());
    }

    #[test]
    fn the_pool_round_trips_through_config() {
        let g = GdnStatePool::qwen38_27b(8, Speculation::DSpark { n: 7 })
            .with_budget(recipe_budget());
        let back: GdnStatePool = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(back, g);
    }
}
