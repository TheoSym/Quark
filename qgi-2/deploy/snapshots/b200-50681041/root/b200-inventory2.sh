set -u
O=/root/qgi2-src2; rm -rf $O; mkdir -p $O/deploy $O/venvs $O/sglang-src $O/supervisor-utils
ls -la /root/qgi2-deploy > $O/deploy.ls 2>&1
find /root/qgi2-deploy -maxdepth 2 -type f \( -name '*.sh' -o -name '*.py' -o -name '*.md' -o -name '*.txt' -o -name '*.json' -o -name '*.yaml' -o -name '*.toml' -o -name '*.conf' -o -name '*.env' \) -size -2M -exec cp --parents {} $O/deploy/ \;
cp -r /opt/supervisor-scripts/utils $O/supervisor-utils/ 2>/dev/null
for v in /venv/sglang /venv/sglang-main; do n=$(basename $v); $v/bin/python -m pip freeze > $O/venvs/$n.freeze.txt 2>&1; $v/bin/python -c "import sys,torch;print(sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda)" > $O/venvs/$n.versions.txt 2>&1; $v/bin/python -c "import sglang,os;print(sglang.__version__, os.path.dirname(sglang.__file__))" >> $O/venvs/$n.versions.txt 2>&1; cat $v/pyvenv.cfg >> $O/venvs/$n.versions.txt; done
ls /venv > $O/venvs/venv.ls
if [ -d /root/sglang/.git ]; then cd /root/sglang; git remote -v > $O/sglang-src/remote.txt; git log -1 --format='%H %ci %s' > $O/sglang-src/head.txt; git status --short > $O/sglang-src/status.txt; git diff > $O/sglang-src/local.patch; fi
ls -d /root/*/ > $O/root-dirs.txt
ls /workspace/.hf_home/hub 2>/dev/null > $O/hf-hub.ls
for f in /models/*.fetch.log; do echo "== $f"; grep -oE '[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+' $f | grep -vE '^(models|root|workspace|venv|usr|opt|tmp)/' | sort | uniq -c | sort -rn | head -3; done > $O/fetch-repos.txt
grep -rhoE '(hf download|snapshot_download|huggingface-cli download|repo_id=)[^;&|]*' /root/qgi2-deploy 2>/dev/null | sort -u > $O/download-cmds.txt
tar czf /root/qgi2-src2.tgz -C /root qgi2-src2 && ls -l /root/qgi2-src2.tgz
