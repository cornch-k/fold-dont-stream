#!/usr/bin/env python3
"""llama 실제 모델 버퍼 = GGUF 전체 텐서 바이트 − 로더가 건너뛴 텐서 바이트.
로그의 "unused tensor NAME (size = N bytes) -- ignoring" 을 합산한다. KV/compute 는 별도."""
import sys, re, os
sys.path.insert(0, 'nanfix/gguf-py')
from gguf import GGUFReader

def tensor_bytes(path):
    r = GGUFReader(path)
    return sum(int(t.n_bytes) for t in r.tensors)

TOT = {}
for log in sys.argv[1:]:
    m = re.search(r'-(A6h16|B6h32|C7h16|D7h32|E8h16)-', log)
    tag = m.group(1) if m else os.path.basename(log)
    model = 'models/qwen35-v2a-hotP.gguf' if 'h32' in tag else 'models/qwen35-v2a-hot.gguf'
    if model not in TOT:
        TOT[model] = tensor_bytes(model)
    skipped = 0
    with open(log, 'rb') as f:
        for line in f:
            mm = re.search(rb'unused tensor .* \(size = (\d+) bytes\)', line)
            if mm:
                skipped += int(mm.group(1))
    loaded = TOT[model] - skipped
    print(f"{tag:6s} 모델버퍼 {loaded/2**30:6.2f} GiB  (건너뜀 {skipped/2**30:5.2f} / 전체 {TOT[model]/2**30:5.2f})")
