#!/usr/bin/env python3
"""우연 바닥이 공정한가: 전문가 출력 행렬의 유효 랭크를 직접 잰다.
lstsq 는 col(A) 로의 직교투영이므로, 공정한 널은 256 이 아니라 rank(A) 로 정해진다."""
import numpy as np, moefit as M
from ceiling import all_expert_outputs
S='/private/tmp/claude-501/-Users-angigyeom-Desktop-DEV-optillama/525ff5d7-3e78-403d-9ac0-15e30e1a2ad7/scratchpad'
N=3072
X=np.fromfile(f'{S}/a_fold/ffn_norm-24_(reshaped).f32',dtype=np.float32)
X=X[:(X.size//N)*N].reshape(-1,N)[:8]
m=M.Model()
E=all_expert_outputs(m,22,X)          # (T,256,3072)
print(f"{'tok':>3} {'cond':>8} {'99%랭크':>8} {'95%랭크':>8} {'s_min/s_max':>12} {'공정바닥(99%)':>13}")
for t in range(X.shape[0]):
    A=E[t].T                           # (3072,256)
    s=np.linalg.svd(A,compute_uv=False)
    en=np.cumsum(s**2)/np.sum(s**2)
    r99=int(np.searchsorted(en,0.99)+1); r95=int(np.searchsorted(en,0.95)+1)
    print(f'{t:3d} {s[0]/s[-1]:8.2f} {r99:8d} {r95:8d} {s[-1]/s[0]:12.4f} {np.sqrt(1-r99/N):13.4f}')
