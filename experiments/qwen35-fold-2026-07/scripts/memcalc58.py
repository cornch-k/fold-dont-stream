#!/usr/bin/env python3
import sys, re
sys.path.insert(0, 'nanfix/gguf-py')
from gguf import GGUFReader
log, model = sys.argv[1], sys.argv[2]
tot = sum(int(t.n_bytes) for t in GGUFReader(model).tensors)
sk = 0
with open(log, 'rb') as f:
    for line in f:
        m = re.search(rb'unused tensor .* \(size = (\d+) bytes\)', line)
        if m: sk += int(m.group(1))
print(f"모델버퍼={(tot-sk)/2**30:.2f}GiB")
