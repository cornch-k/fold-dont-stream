#!/usr/bin/env python3
"""층 0 선형 어텐션 재귀부 후보 스윕. ref attn_out = attn_residual-0 − input_embed."""
import glob, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_update
import fold_model

IDS = [760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388]
N = len(IDS)


def ref(name, d):
    for f in sorted(glob.glob(f"heal/ref/{name}_*.bin")):
        a = np.fromfile(f, dtype=np.float32)
        if a.size == N * d:
            yield a.reshape(N, d)


def rel(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))


def np32(x):
    return np.array(x.astype(mx.float32))


model, args, cfg = fold_model.load()
model.eval()
x = mx.array([IDS])
h = model.embed(x)
emb = np32(h[0])
resid = list(ref("attn_residual-0", 3072))
tgt = [r - emb for r in resid]
print(f"ref attn_out 노름: {[round(float(np.linalg.norm(t)),1) for t in tgt]}  (emb 노름 {np.linalg.norm(emb):.1f})")

m = model.layers[0].linear_attn
a_in = model.layers[0].input_layernorm(h)
B, S = 1, N
qkv = m.in_proj_qkv(a_in)
z = m.in_proj_z(a_in).reshape(B, S, m.num_v_heads, m.head_v_dim)
bb = m.in_proj_b(a_in)
aa = m.in_proj_a(a_in)
cs = mx.zeros((B, m.conv_kernel_size - 1, m.conv_dim), dtype=a_in.dtype)
conv = nn.silu(m.conv1d(mx.concatenate([cs, qkv], axis=1)))
q0, k0, v0 = [t.reshape(B, S, hh, dd) for t, hh, dd in zip(
    mx.split(conv, [m.key_dim, 2 * m.key_dim], -1),
    [m.num_k_heads, m.num_k_heads, m.num_v_heads],
    [m.head_k_dim, m.head_k_dim, m.head_v_dim])]

D = m.head_k_dim
inv = D ** -0.5


def run(qs, ks, av, bv, tag):
    out, _ = gated_delta_update(qs, ks, v0, av, bv, m.A_log, m.dt_bias, None, None, use_kernel=True)
    o = m.norm(out, z)
    o = m.out_proj(o.reshape(B, S, -1))
    mx.eval(o)
    mine = np32(o[0])
    r = min(rel(t, mine) for t in tgt)
    print(f"  {tag:34s} rel={r:.4e}  노름 {np.linalg.norm(mine):8.1f}")
    return r


rq, rk = mx.fast.rms_norm(q0, None, 1e-6), mx.fast.rms_norm(k0, None, 1e-6)
print("── 후보 ──")
run(inv * inv * rq, inv * rk, aa, bb, "MLX 기본 (q/√d·l2, k l2)")
run(inv * rq, inv * rk, aa, bb, "q l2 (llama.cpp 규약)")
run(inv * inv * rq, inv * inv * rk, aa, bb, "q,k 모두 /√d")
run(inv * rq, rk, aa, bb, "q l2, k ×√d")
run(inv * inv * rq, inv * rk, bb, aa, "기본 + a/b 교환")
run(inv * inv * inv * rq, inv * rk, aa, bb, "q 에 1/√d 추가 (=l2/d)")
run(inv * inv * rq / D, inv * rk, aa, bb, "q 에 1/d 추가")
