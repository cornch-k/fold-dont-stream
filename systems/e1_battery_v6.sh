#!/bin/zsh
# E1 배터리 v5 — llama-cli(-nr) + 워치독 + 스왑가드 + peakRSS ('status'는 zsh 예약어라 st 사용)
cd /Users/angigyeom/Desktop/optillama/llama.cpp
LM=../models/Llama-3.3-70B-GGUF/Llama-3.3-70B-Instruct-Q4_K_M.gguf
LG=../models/Laguna-S-2.1-GGUF/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf
RES=/Users/angigyeom/Desktop/optillama/protolab/e1_results.log
# append 모드 (llama_fold 결과 보존)

swap_mb() { sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); printf "%d", $6}'; }
pageins() { vm_stat | awk '/Pageins/{gsub(/\./,"");print $2}'; }

run() {
  local name=$1 fold=$2 model=$3 ngl=$4 ntok=$5 maxsec=$6
  sync; sleep 3
  local pi0=$(pageins)
  local sw0=$(swap_mb)
  local t0=$(date +%s)
  local peak=0
  local st=TIMEOUT
  local LOG=/tmp/e1v5_$name.log
  if [ -n "$fold" ]; then
    LLAMA_FOLD_K=$fold LLAMA_MMAP_NO_PREFETCH=1 \
      build-m1max/bin/llama-cli -m $model -p "The story begins" -n $ntok -t 8 -fa on \
      -ngl $ngl --no-warmup -no-cnv --no-repack < /dev/null > $LOG 2>&1 &
  else
    LLAMA_MMAP_NO_PREFETCH=1 \
      build-m1max/bin/llama-cli -m $model -p "The story begins" -n $ntok -t 8 -fa on \
      -ngl $ngl --no-warmup -no-cnv --no-repack < /dev/null > $LOG 2>&1 &
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
  echo "$name st=$st elapsed=$(( $(date +%s) - t0 ))s peakRSS=$((peak/1048576))GB ssd_read=${gb}GB swap_delta=$(( $(swap_mb) - sw0 ))MB $perf" >> $RES
}

#   이름              fold  모델  ngl  n   제한
# (완료) llama_fold
# (금지) metal — 원인 규명 전까지
run laguna_fold       16   $LG   0    32  900
run llama_base        ""   $LM   0    8   2700
run laguna_base       ""   $LG   0    8   2700
echo BATTERY_DONE >> $RES
