#!/bin/bash
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
echo $$ > exp/runchain6.pid
for p in exp/runchain.pid exp/runchain2.pid exp/runchain3.pid exp/runchain4.pid exp/runchain5.pid; do
  while [ -f "$p" ] && kill -0 "$(cat $p 2>/dev/null)" 2>/dev/null; do sleep 30; done
done
bash exp/exp67.sh > exp/exp67.out 2>&1
rm -f exp/runchain6.pid
