set -u
O=/root/qgi2-final; rm -rf $O; mkdir -p $O/deploy $O/supervisor $O/etc $O/venvs $O/caches $O/models
cp /root/qgi2-deploy/* $O/deploy/ 2>/dev/null
cp /etc/supervisor/conf.d/qgi2-*.conf /opt/supervisor-scripts/qgi2-*.sh $O/supervisor/
cp /etc/portal.yaml $O/etc/; cp /etc/supervisor/supervisord.conf $O/etc/ 2>/dev/null; supervisorctl status > $O/etc/supervisor-status.txt
tailscale serve status > $O/etc/tailscale-serve.txt 2>&1; tailscale status --self 2>/dev/null | head -2 >> $O/etc/tailscale-serve.txt
for v in /venv/sglang /venv/sglang-main /venv/vllm; do n=$(basename $v); $v/bin/python - > $O/venvs/$n.freeze.txt <<'EOF'
import importlib.metadata as m
for d in sorted(m.distributions(), key=lambda d: d.metadata["Name"].lower()): print(f'{d.metadata["Name"]}=={d.version}')
EOF
done
for r in /root/sglang /root/sglang-main; do (cd $r && echo "$(basename $r) $(git rev-parse HEAD 2>/dev/null || cat .git 2>/dev/null)"; git diff --stat 2>/dev/null) >> $O/venvs/sglang-commits.txt; done
/venv/sglang-main/bin/python -c "import sglang,os;print('sglang-main path', os.path.dirname(sglang.__file__))" >> $O/venvs/sglang-commits.txt 2>&1
cat /root/sglang-main/.git 2>/dev/null >> $O/venvs/sglang-commits.txt; (cd /root/sglang-main 2>/dev/null && git log -1 --format='%H %ci %s' >> $O/venvs/sglang-commits.txt 2>&1)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1 > $O/etc/host.txt; nvcc --version | tail -1 >> $O/etc/host.txt; cat /etc/os-release | head -2 >> $O/etc/host.txt; free -g | head -2 >> $O/etc/host.txt
# caches worth carrying (small): flashinfer autotune + vllm torch.compile cache index
tar czf $O/caches/flashinfer-autotune.tgz -C /root/.cache/sglang flashinfer 2>/dev/null; du -sh /root/.cache/sglang /root/.cache/flashinfer /root/.cache/vllm 2>/dev/null > $O/caches/sizes.txt
# weights on disk: repo/revision from pins or HF metadata, sizes, sha of config.json
for d in /models/*/; do n=$(basename $d); echo "== $n $(du -sh $d | cut -f1) files=$(find $d -type f | wc -l)"; cat $d/.qgi2-pin.json 2>/dev/null; [ -f $d/config.json ] && sha256sum $d/config.json | cut -c1-16; done > $O/models/inventory.txt
df -h / | tail -1 >> $O/models/inventory.txt
tar czf /root/qgi2-final.tgz -C /root qgi2-final && ls -l /root/qgi2-final.tgz
