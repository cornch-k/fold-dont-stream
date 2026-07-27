#!/usr/bin/env python3
"""B1 독립 검증: 깊은 층 천장이 정말 41에서만 우연 바닥인가.
리뷰어 스크립트를 쓰지 않고 ceiling.py 의 원래 코드 경로로 다시 푼다."""
import sys, time, numpy as np, moefit as M
from ceiling import all_expert_outputs, oracle_resid

S='/private/tmp/claude-501/-Users-angigyeom-Desktop-DEV-optillama/525ff5d7-3e78-403d-9ac0-15e30e1a2ad7/scratchpad'
N=3072
L=48
WL=[0]+[1+((i-1)//3)*3 for i in range(1,L)]
for v in (45,46,47): WL[v]=v

def load(d, il, n):
    a=np.fromfile(f'{S}/{d}/ffn_norm-{il}_(reshaped).f32', dtype=np.float32)
    a=a[:(a.size//N)*N].reshape(-1,N)
    return a[:n]

T=int(sys.argv[1]) if len(sys.argv)>1 else 128
ils=[int(x) for x in sys.argv[2].split(',')]
m=M.Model()
floor=np.sqrt(1-256/3072)
print(f'T={T}  우연 바닥 {floor:.4f}\n')
print(f"{'il':>3} {'wl':>3} {'접힌X':>8} {'무접기X':>8} {'에너지%':>8} {'바닥대비':>8}")
for il in ils:
    wl=WL[il]
    row=[]
    for d in ('a_fold','a_unf'):
        X=load(d,il,T)
        R=M.ffn_block(m,il,X)-M.shexp_out(m,il,X)
        E=all_expert_outputs(m,wl,X); c=oracle_resid(E,R); del E
        row.append(c)
    en=(1-row[1]**2)*100
    print(f'{il:3d} {wl:3d} {row[0]:8.4f} {row[1]:8.4f} {en:7.1f}% {en/8.333:7.2f}x', flush=True)
