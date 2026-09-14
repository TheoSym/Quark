set -u
O=/root/qgi2-src3; rm -rf $O; mkdir -p $O
for v in /venv/sglang /venv/sglang-main; do n=$(basename $v)
  $v/bin/python - > $O/$n.freeze.txt <<'EOF'
import importlib.metadata as m
for d in sorted(m.distributions(), key=lambda d: d.metadata["Name"].lower()):
    print(f'{d.metadata["Name"]}=={d.version}')
EOF
  $v/bin/python -c "import sglang,os,subprocess;p=os.path.dirname(sglang.__file__);print(p);print(open(os.path.join(p,'version.py')).read().strip() if os.path.exists(os.path.join(p,'version.py')) else '')" > $O/$n.sglang-path.txt 2>&1
done
for r in /root/sglang /root/sglang-main; do n=$(basename $r); [ -d $r/.git ] || continue; (cd $r; git remote get-url origin; git rev-parse HEAD; git log -1 --format='%ci %s'; git branch --show-current; git status --short) > $O/$n.git.txt 2>&1; (cd $r; git diff) > $O/$n.local.patch; done
for d in /models/*/; do n=$(basename $d); echo "== $n"; head -30 $d/README.md 2>/dev/null | grep -iE 'base_model|model_name|repo|^# |license:' | head -5; cat $d/.qgi2-pin.json 2>/dev/null; done > $O/model-readmes.txt
grep -rhoE '(RadixArk|nvidia|meta|Qwen|deepseek-ai|incoai|SamSammane|syv-ai)/[A-Za-z0-9._-]+' /root/qgi2-deploy /opt/supervisor-scripts/qgi2-*.sh /root/*.sh 2>/dev/null | sort | uniq -c | sort -rn > $O/repo-mentions.txt
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/gpu-apps.csv
tar czf /root/qgi2-src3.tgz -C /root qgi2-src3 && echo ok
