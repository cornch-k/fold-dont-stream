#!/bin/zsh
cd /Users/angigyeom/Desktop/optillama/protolab/nanbeige-llamacpp
M1=../../models/Nanbeige4.2-3B-GGUF/nanbeige4.2-3b-Q8_0.gguf
M2=../../models/Nanbeige4.2-3B-GGUF/nanbeige4.2-3b-Q8_0-flat44.gguf
RES=/Users/angigyeom/Desktop/optillama/protolab/twin_results.log
: >> $RES
FREE_GB=$(/opt/anaconda3/bin/python3 -c "
import subprocess
out = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
d = {}
for line in out.splitlines():
    if ':' not in line: continue
    k, v = line.split(':', 1)
    try: d[k.strip()] = int(v.strip().rstrip('.'))
    except ValueError: pass
print(round((d.get('Pages free',0)+d.get('Pages inactive',0)+d.get('Pages speculative',0))*16384/1e9,2))")
BALLAST_GB=$(/opt/anaconda3/bin/python3 -c "print(max(0, int($FREE_GB - 6.5 - 1)))")
echo "free-like=${FREE_GB}GB ballast=${BALLAST_GB}GB" >> $RES
/opt/anaconda3/bin/python3 ../ballast.py --gb $BALLAST_GB > /tmp/ballast_run.log 2>&1 &
BPID=$!
sleep 20
run() {
  local name=$1 env_stride=$2 model=$3
  sync; sleep 3
  local pi0=$(vm_stat | awk '/Pageins/{gsub(/\./,"");print $2}')
  local sw0=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); printf "%d", $6}')
  local t0=$(date +%s) peak=0 st=TIMEOUT
  local LOG=/tmp/twin_run_${name}.log
  if [ -n "$env_stride" ]; then
    NANBEIGE_BOUNDARY_STRIDE=$env_stride LLAMA_MMAP_NO_PREFETCH=1 build/bin/llama-cli -m $model -p "The story begins" -n 32 -t 8 -fa on -ngl 0 --no-warmup -no-cnv -nr < /dev/null > $LOG 2>&1 &
  else
    LLAMA_MMAP_NO_PREFETCH=1 build/bin/llama-cli -m $model -p "The story begins" -n 32 -t 8 -fa on -ngl 0 --no-warmup -no-cnv -nr < /dev/null > $LOG 2>&1 &
  fi
  local PID=$!
  while true; do
    sleep 5
    local rss=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    [ -z "$rss" ] && { st=EXITED; break; }
    [ "$rss" -gt "$peak" ] && peak=$rss
    grep -q '\[ Prompt:' $LOG 2>/dev/null && { st=OK; kill -9 $PID 2>/dev/null; break; }
    local swnow=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); printf "%d", $6}')
    [ $((swnow - sw0)) -gt 1500 ] && { st=SWAP_ABORT; kill -9 $PID 2>/dev/null; break; }
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { kill -9 $PID 2>/dev/null; break; }
  done
  local perf=$(grep -o '\[ Prompt:.*\]' $LOG | tail -1)
  local gb=$(( ( $(vm_stat | awk '/Pageins/{gsub(/\./,"");print $2}') - pi0 ) * 16384 / 1073741824 ))
  echo "$name st=$st elapsed=$(( $(date +%s) - t0 ))s peakRSS=$((peak/1048576))GB ssd_read=${gb}GB swap_d=$(( $(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); printf "%d", $6}') - sw0 ))MB $perf" >> $RES
}
run folded ""  $M1
run flat   22  $M2
kill -TERM $BPID 2>/dev/null
echo TWIN_DONE >> $RES
