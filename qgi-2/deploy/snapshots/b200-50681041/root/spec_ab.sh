#!/bin/bash
# Head-to-head: same 6 prompts, greedy, thinking off, 400 tokens, sequential, per port.
P1="Write a Python function that parses ISO-8601 dates and returns a datetime, with docstring and 3 tests."
P2="Explain in 200 words how a hash map handles collisions."
P3="Write a bash script that rotates logs in /var/log/app older than 7 days."
P4="Summarize the plot of Hamlet in 150 words."
P5="Implement an LRU cache class in Python with get and put, O(1) each, with a short usage example."
P6="Write a SQL query that returns the top 3 customers by total order value per country, and explain it."
for spec in "18035 Qwen3.8-Flash-Next-NVFP4 nvidia" "18036 Qwen3.8-Flash-Next-NVFP4-RadixArk radixark"; do
  set -- $spec; port=$1; model=$2; label=$3; B=http://127.0.0.1:$port
  curl -s $B/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":30}" >/dev/null
  tot=0; t0=$(date +%s.%N)
  for p in "$P1" "$P2" "$P3" "$P4" "$P5" "$P6"; do
    n=$(curl -s $B/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":400,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | python3 -c "import sys,json;print(json.load(sys.stdin)[\"usage\"][\"completion_tokens\"])")
    tot=$((tot+n))
  done
  t1=$(date +%s.%N)
  acc=$(curl -s $B/metrics | grep "^sglang:spec_accept_length" | grep -o "[0-9.]*$")
  rate=$(curl -s $B/metrics | grep "^sglang:spec_accept_rate" | grep -o "[0-9.]*$")
  echo "$label: tokens=$tot secs=$(python3 -c "print(round($t1-$t0,1))") tok_s=$(python3 -c "print(round($tot/($t1-$t0),1))") accept_len=$acc accept_rate=$rate"
done
