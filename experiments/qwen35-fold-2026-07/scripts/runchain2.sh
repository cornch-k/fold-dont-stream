#!/bin/bash
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
echo $$ > exp/runchain2.pid
while [ -f exp/runchain.pid ] && kill -0 "$(cat exp/runchain.pid 2>/dev/null)" 2>/dev/null; do sleep 30; done
bash exp/exp63.sh > exp/exp63.out 2>&1
rm -f exp/runchain2.pid
