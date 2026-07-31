#!/bin/bash
# EXP-61 이 끝나면 EXP-62 를 이어서 실행
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
while pgrep -f exp61.sh >/dev/null 2>&1; do sleep 20; done
bash exp/exp62.sh > exp/exp62.out 2>&1
