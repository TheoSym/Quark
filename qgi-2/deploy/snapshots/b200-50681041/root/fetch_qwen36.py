from huggingface_hub import snapshot_download
for repo, dest in [("Qwen/Qwen3.6-35B-A3B-FP8", "/models/Qwen3.6-35B-A3B-FP8"), ("z-lab/Qwen3.6-35B-A3B-DFlash", "/models/Qwen3.6-35B-A3B-DFlash")]:
    print("==", repo, flush=True); snapshot_download(repo, local_dir=dest, max_workers=16)
print("FETCH_DONE", flush=True)
