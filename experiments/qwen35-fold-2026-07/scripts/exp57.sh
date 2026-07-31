#!/bin/bash
# EXP-57 용량 재협상: 남은 속도 예산(39.9 vs 제약 25)을 정확도로 바꾼다.
# 판정 지표를 wired(시스템 전체) 대신 llama 프로세스 최대 RSS 로 교체.
set -u
KILL_GB=24
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
PPL=nanfix/build/bin/llama-perplexity
CLI=nanfix/build/bin/llama-cli
WIKI=nanfix/scripts/wikitext-2-raw/wiki.test.raw
HOT16=models/qwen35-v2a-hot.gguf
HOT32=models/qwen35-v2a-hotP.gguf
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

mkmap() { # $1 = rep 수 → MAP, GAMMA 파일 생성
  local G=$1 F=44 il g; local -a m
  for ((il=0; il<48; il++)); do
    if (( il >= F )); then m[il]=$il
    else g=$(( il * G / F )); m[il]=$(( (g * F + G - 1) / G )); fi
  done
  local IFS=,; MAP="${m[*]}"; unset IFS
  : > $EXP/g57-$G.txt
  for ((il=0; il<48; il++)); do
    if [ "${m[il]}" = "$il" ]; then echo "0.3"
    elif (( il < 23 )); then echo "0.0"
    else echo "0.3"; fi
  done >> $EXP/g57-$G.txt
  GAM=$EXP/g57-$G.txt
  UNIQ=$(echo "$MAP" | tr ',' '\n' | sort -un | tr '\n' ' ')
}

settle() { local s=0; while (( s < 36 )); do local W=$(wired_gb); awk -v w="$W" 'BEGIN{exit !(w<9)}' && return; sleep 5; s=$((s+1)); done; }

# $1=pid $2=maxiter → "maxRSS_GB peakWired_GB"
watch() {
  local PID=$1 MAXI=$2 i=0 RS=0 PW=0 W R
  while kill -0 $PID 2>/dev/null && (( i < MAXI )); do
    R=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    [ -n "$R" ] && awk -v r="$R" -v m="$RS" 'BEGIN{exit !(r/1048576>m)}' && RS=$(awk -v r="$R" 'BEGIN{printf "%.2f", r/1048576}')
    W=$(wired_gb); awk -v w="$W" -v p="$PW" 'BEGIN{exit !(w>p)}' && PW=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "$RS $PW"
}

P='The history of computing began with mechanical calculators. Charles Babbage designed the Analytical Engine in the 1830s, and Ada Lovelace wrote algorithms for it. In the twentieth century, electronic computers replaced mechanical ones. The story continues:'

run() { # $1=tag $2=model $3=rep수
  local TAG=$1 M=$2 G=$3
  mkmap $G
  echo "══ $TAG: rep=$G 유일층 [$UNIQ] 모델 $(basename $M)" 
  local E="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$GAM LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1"
  settle
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-57 $TAG ppl" >> $EXP/timeline.log
  env $E $PPL -m $M -f $WIKI --chunks 16 -c 512 --no-repack -b 512 -ngl 99 \
      --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e57-$TAG-ppl.log 2>&1 &
  read RS PW <<< "$(watch $! 72)"
  local PPLV=$(grep -a 'Final estimate' $EXP/e57-$TAG-ppl.log | tail -1 | sed 's/.*PPL = //')
  settle
  env $E $CLI -m $M -p "$P" -n 300 --ignore-eos -ngl 99 --no-repack -no-cnv -st --no-warmup \
      -c 1024 --seed 1337 --no-jinja --temp 0 --repeat-penalty 1.15 --logit-bias 248068-20 \
      --spec-type ngram-simple --spec-draft-n-max 12 \
      --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e57-$TAG-spd.log 2>&1 < /dev/null &
  read RS2 PW2 <<< "$(watch $! 84)"
  local SPD=$(grep -ao 'Generation: [0-9.]* t/s' $EXP/e57-$TAG-spd.log | tail -1 | sed 's/Generation: //')
  local MAXRSS=$(awk -v a="$RS" -v b="$RS2" 'BEGIN{print (a>b)?a:b}')
  echo "RESULT $TAG  ppl=$PPLV  속도=$SPD  llamaRSS=${MAXRSS}GB  wired=$PW2"
}

run A6h16 $HOT16 6
run B6h32 $HOT32 6
run C7h16 $HOT16 7
run D7h32 $HOT32 7
run E8h16 $HOT16 8
echo "EXP-57 완료"
