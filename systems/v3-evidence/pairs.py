#!/usr/bin/env python3
"""인접 짝이 최선인가? 층 24 를 여러 물리층으로 접었을 때의 천장을 비교."""
import numpy as np, time, sys
import moefit as M
from ceiling import load_folded_X, all_expert_outputs, oracle_resid

il = 24
m = M.Model()
X = load_folded_X(il, 64)
Y = M.ffn_block(m, il, X); Sh = M.shexp_out(m, il, X)
R = Y - Sh
print(f'층 {il} 목표 대비 천장 (낮을수록 표현 가능)\n')
print(f"{'wl':>3} {'거리':>4} {'천장':>7}")
for wl in (23, 25, 22, 26, 20, 12, 40):
    t0 = time.time()
    E = all_expert_outputs(m, wl, X)
    c = oracle_resid(E, R); del E
    print(f'{wl:3d} {abs(wl-il):4d} {c:7.3f}   [{time.time()-t0:.0f}s]', flush=True)
