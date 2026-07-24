#!/bin/zsh
# 야간 측정 공용 워치독 (e1_battery.sh v5 run() 기반, 'status' 예약어 회피 -> st)
cd /Users/angigyeom/Desktop/optillama/llama.cpp

swap_mb() { sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); printf "%d", $6}'; }
pageins() { vm_stat | awk '/Pageins/{gsub(/\./,"");print $2}'; }

# run <name> <fold_k or ""> <model_path> <ngl> <ntok> <maxsec> <extra_args...>
run() {
  local name=$1 fold=$2 model=$3 ngl=$4 ntok=$5 maxsec=$6
  shift 6
  local extra_args=("$@")
  sync; sleep 3
  local pi0=$(pageins)
  local sw0=$(swap_mb)
  local t0=$(date +%s)
  local peak=0
  local st=TIMEOUT
  local LOG=/tmp/night_ops/${name}.log
  : > $LOG
  if [ -n "$fold" ]; then
    LLAMA_FOLD_K=$fold LLAMA_MMAP_NO_PREFETCH=1 \
      build-m1max/bin/llama-cli -m $model -p "The story begins" -n $ntok -t 8 -fa on \
      -ngl $ngl --no-warmup -no-cnv --no-repack "${extra_args[@]}" < /dev/null > $LOG 2>&1 &
  else
    LLAMA_MMAP_NO_PREFETCH=1 \
      build-m1max/bin/llama-cli -m $model -p "The story begins" -n $ntok -t 8 -fa on \
      -ngl $ngl --no-warmup -no-cnv --no-repack "${extra_args[@]}" < /dev/null > $LOG 2>&1 &
  fi
  local PID=$!
  while true; do
    sleep 5
    local rss=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    if [ -z "$rss" ]; then st=EXITED; break; fi
    [ "$rss" -gt "$peak" ] && peak=$rss
    if grep -q '\[ Prompt:' $LOG 2>/dev/null; then st=OK; kill -9 $PID 2>/dev/null; break; fi
    if [ $(( $(swap_mb) - sw0 )) -gt 1500 ]; then st=SWAP_ABORT; kill -9 $PID 2>/dev/null; break; fi
    if [ $(( $(date +%s) - t0 )) -gt $maxsec ]; then st=TIMEOUT; kill -9 $PID 2>/dev/null; break; fi
  done
  sleep 2
  local perf=$(grep -o '\[ Prompt:.*\]' $LOG | tail -1)
  local gb=$(( ( $(pageins) - pi0 ) * 16384 / 1073741824 ))
  local elapsed=$(( $(date +%s) - t0 ))
  local peakgb=$((peak/1048576))
  local swapd=$(( $(swap_mb) - sw0 ))
  echo "RESULT name=$name st=$st elapsed=${elapsed}s peakRSS=${peakgb}GB ssd_read=${gb}GB swap_delta=${swapd}MB $perf log=$LOG"
}
