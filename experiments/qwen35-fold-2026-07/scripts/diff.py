#!/usr/bin/env python3
"""llama.cpp 저장 활성값 vs MLX 활성값 층별 대조. 첫 발산 층을 찾는다."""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3_5 import create_ssm_mask
import fold_model

PROMPT_IDS = [760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388]
REF = "heal/ref"


def load_ref(name):
    """이름이 같은 파일이 여러 개면(웜업+본실행) 카운터 큰 쪽을 쓴다."""
    fs = sorted(glob.glob(f"{REF}/{name}_*.bin"),
                key=lambda p: int(re.search(r"_(\d+)\.bin$", p).group(1)))
    if not fs:
        return None
    out = []
    for f in fs:
        a = np.fromfile(f, dtype=np.float32)
        if a.size % 3072:
            continue
        a = a.reshape(-1, 3072)
        if a.shape[0] == len(PROMPT_IDS):   # 프롬프트 일괄 처리 패스만 (디코드 패스 제외)
            out.append(a)
    return out


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))


model, args, cfg = fold_model.load()
model.eval()
x = mx.array([PROMPT_IDS])

# 임베딩
emb = np.array(model.embed(x)[0].astype(mx.float32))
for i, r in enumerate(load_ref("model.input_embed") or []):
    print(f"input_embed[{i}] ref{r.shape} vs mlx{emb.shape}  rel={rel(r[-emb.shape[0]:], emb):.3e}")

# 층별
h = model.embed(x)
cache = [None] * len(model.layers)
fa_mask = create_attention_mask(h, None)
ssm_mask = create_ssm_mask(h, None)
worst = None
for il, layer in enumerate(model.layers):
    a = layer.input_layernorm(h)
    attn = layer.linear_attn if layer.is_linear else layer.self_attn
    hh = h + attn(a, (ssm_mask if layer.is_linear else fa_mask), None)
    mx.eval(hh)
    refs = load_ref(f"attn_residual-{il}") or []
    ds = [rel(r[-hh.shape[1]:], np.array(hh[0].astype(mx.float32))) for r in refs]
    h = hh + layer.mlp(layer.post_attention_layernorm(hh))
    mx.eval(h)
    refo = load_ref(f"l_out-{il}") or []
    do = [rel(r[-h.shape[1]:], np.array(h[0].astype(mx.float32))) for r in refo]
    kind = "lin" if layer.is_linear else "FULL"
    print(f"층 {il:2d} {kind}  attn_res rel={min(ds) if ds else float('nan'):.3e}"
          f"   l_out rel={min(do) if do else float('nan'):.3e}", flush=True)
    if worst is None and ds and min(ds) > 0.05:
        worst = il
        print(f"  ↑ 첫 발산: 층 {il} 어텐션 분기")
    mx.clear_cache()
