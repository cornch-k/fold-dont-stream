#!/bin/bash
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
echo $$ > exp/runchain3.pid
for p in exp/runchain.pid exp/runchain2.pid; do
  while [ -f "$p" ] && kill -0 "$(cat $p 2>/dev/null)" 2>/dev/null; do sleep 30; done
done
bash exp/exp64.sh > exp/exp64.out 2>&1
rm -f exp/runchain3.pid
