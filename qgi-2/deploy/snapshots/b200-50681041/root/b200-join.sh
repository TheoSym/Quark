set -e
tailscale up --reset --authkey "file:/root/ts.key" --hostname qgi2-g720 --accept-routes --accept-dns=false --advertise-tags=tag:gpu-lane 2>&1 | tail -2
shred -u /root/ts.key 2>/dev/null || rm -f /root/ts.key
for p in 18034 18035 18037; do tailscale serve --bg --tcp $p tcp://127.0.0.1:$p >/dev/null; done
tailscale ip -4
tailscale serve status
