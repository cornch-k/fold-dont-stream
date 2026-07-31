#!/usr/bin/env python3
"""어텐션 내부 단계별 대조. 층 0(linear)·층 3(full) 을 손으로 재현해 첫 발산 연산을 찍는다."""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3_5 import create_ssm_mask
import fold_model

IDS = [760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388]
N = len(IDS)
REF = "heal/ref"


def ref(name, dim=None):
    """ne[0]=dim 인 프롬프트 패스 텐서. dim 미지정이면 총 원소수/N 로 추론."""
    best = []
    for f in sorted(glob.glob(f"{REF}/{name}_*.bin")):
        a = np.fromfile(f, dtype=np.float32)
        d = dim or (a.size // N)
        if a.size == N * d:
            best.append(a.reshape(N, d))
    return best


def rel(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    if a.size != b.size:
        return float('nan')
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))


def np32(x):
    return np.array(x.astype(mx.float32))


def show(tag, name, mine, dim=None):
    rs = ref(name, dim)
    if not rs:
        print(f"  {tag:22s} ref 없음"); return
    print(f"  {tag:22s} rel={min(rel(r, mine) for r in rs):.3e}   ref{rs[0].shape} mine{np.shape(mine)}")


model, args, cfg = fold_model.load()
model.eval()
x = mx.array([IDS])
h = model.embed(x)
show("input_embed", "model.input_embed", np32(h[0]))

fa_mask = create_attention_mask(h, None)
ssm_mask = create_ssm_mask(h, None)

for il in (0, 3):
    layer = model.layers[il]
    a = layer.input_layernorm(h if il == 0 else h)
    print(f"── 층 {il} ({'linear' if layer.is_linear else 'FULL'}) ──")
    if layer.is_linear:
        m = layer.linear_attn
        qkv = m.in_proj_qkv(a)
        show("qkv_mixed", f"linear_attn_qkv_mixed-{il}", np32(qkv[0]))
        z = m.in_proj_z(a)
        show("z", f"z-{il}", np32(z[0]))
        cs = mx.zeros((1, m.conv_kernel_size - 1, m.conv_dim), dtype=a.dtype)
        conv_out = nn.silu(m.conv1d(mx.concatenate([cs, qkv], axis=1)))
        show("conv_output_silu", f"conv_output_silu-{il}", np32(conv_out[0]))
        out = m(a, ssm_mask, None)
        show("attn_output(=out_proj)", f"attn_output-{il}", np32(out[0]))
    else:
        m = layer.self_attn
        qf = m.q_proj(a)
        show("Qcur_full", f"Qcur_full-{il}", np32(qf[0]))
        B, L, _ = a.shape
        q, g = mx.split(qf.reshape(B, L, m.num_attention_heads, -1), 2, axis=-1)
        show("Qcur_normed", f"Qcur_normed-{il}", np32(m.q_norm(q)[0]))
        k = m.k_norm(m.k_proj(a).reshape(B, L, m.num_key_value_heads, -1))
        show("Kcur_normed", f"Kcur_normed-{il}", np32(k[0]))
        show("gate_sigmoid", f"gate_sigmoid-{il}", np32(mx.sigmoid(g.reshape(B, L, -1))[0]))
        out = m(a, fa_mask, None)
        show("attn_output(=o_proj)", f"attn_output-{il}", np32(out[0]))
    # 다음 층으로 전진 (층 0~3 순차)
    for j in range(il, il + 1):
        pass
    if il == 0:
        for j in range(0, 3):
            ly = model.layers[j]
            hh = h + (ly.linear_attn if ly.is_linear else ly.self_attn)(
                ly.input_layernorm(h), ssm_mask if ly.is_linear else fa_mask, None)
            h = hh + ly.mlp(ly.post_attention_layernorm(hh))
            mx.eval(h)
