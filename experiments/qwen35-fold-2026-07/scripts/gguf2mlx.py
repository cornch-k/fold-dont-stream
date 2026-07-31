#!/usr/bin/env python3
"""접힌 GGUF(v2a-hot) → MLX safetensors + config. 상주 텐서만 꺼내 접기 결선을 물리적으로 재현.

핵심: 접기는 "논리층 il 이 물리층 wl 의 FFN 가중치를 쓴다"이므로, MLX 쪽에서는 48개 층을
모두 만들고 접힌 층에는 대표층의 전문가 가중치를 **복제**해 넣는다. 학습 시에는 tie 를
유지해야 하므로 mlx_fold.py 가 복제 대신 공유 참조로 재결선한다(여기서는 저장만).

핫 전문가: 접힌 층 il 의 hexps 는 그 층 고유이므로 그대로 별 이름으로 저장하고,
mlx_fold.py 가 δ 가산 분기로 붙인다.
"""
import sys, json, os, argparse
sys.path.insert(0, '/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/gguf-py')
import numpy as np
from gguf import GGUFReader
import gguf.quants as Q
import mlx.core as mx

AP = argparse.ArgumentParser()
AP.add_argument("--gguf", default="/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hot.gguf")
AP.add_argument("--out", default="/Users/angigyeom/Desktop/DEV/fun/accelerator/heal/mlx-fold")
AP.add_argument("--reps", type=int, default=6, help="접힌 그룹 수")
AP.add_argument("--fixfrom", type=int, default=44, help="이 인덱스부터 고정")
AP.add_argument("--bits", type=int, default=4)
A = AP.parse_args()

def build_map(G, F, L=48):
    m = []
    for il in range(L):
        if il >= F: m.append(il)
        else:
            g = il * G // F
            m.append((g * F + G - 1) // G)
    return m

WL = build_map(A.reps, A.fixfrom)
REPS = sorted(set(WL))
print(f"맵: 물리층 {len(REPS)}개 = {REPS}")

r = GGUFReader(A.gguf)
T = {t.name: t for t in r.tensors}
def deq(name):
    t = T[name]
    d = Q.dequantize(t.data, t.tensor_type)
    return np.ascontiguousarray(d.astype(np.float32))

cfg = json.load(open('/Users/angigyeom/Desktop/DEV/fun/accelerator/exp/qwen35-config.json'))
tc = cfg.get('text_config', cfg)
NL = 48   # MTP(48) 제외
out = {}

def put(k, v): out[k] = mx.array(v)

# 임베딩/출력
put("language_model.model.embed_tokens.weight", deq("token_embd.weight"))
put("language_model.model.norm.weight", deq("output_norm.weight"))
put("language_model.lm_head.weight", deq("output.weight"))

for il in range(NL):
    P = f"language_model.model.layers.{il}"
    wl = WL[il]
    is_full = ((il + 1) % tc['full_attention_interval'] == 0)
    put(f"{P}.input_layernorm.weight", deq(f"blk.{il}.attn_norm.weight"))
    put(f"{P}.post_attention_layernorm.weight", deq(f"blk.{wl}.attn_post_norm.weight"))

    if is_full:
        qkv = deq(f"blk.{il}.attn_qkv.weight")           # (out, in) = (n_q+n_k+n_v, hidden)
        nq = tc['num_attention_heads'] * tc['head_dim']
        nk = tc['num_key_value_heads'] * tc['head_dim']
        put(f"{P}.self_attn.q_proj.weight", qkv[:nq])
        put(f"{P}.self_attn.k_proj.weight", qkv[nq:nq+nk])
        put(f"{P}.self_attn.v_proj.weight", qkv[nq+nk:])
        put(f"{P}.self_attn.o_proj.weight", deq(f"blk.{il}.attn_out.weight"))
        put(f"{P}.self_attn.q_norm.weight", deq(f"blk.{il}.attn_q_norm.weight"))
        put(f"{P}.self_attn.k_norm.weight", deq(f"blk.{il}.attn_k_norm.weight"))
    else:
        LA = f"{P}.linear_attn"
        qkv = deq(f"blk.{il}.attn_qkv.weight")
        put(f"{LA}.in_proj_qkv.weight", qkv)
        put(f"{LA}.in_proj_z.weight", deq(f"blk.{il}.attn_gate.weight"))
        put(f"{LA}.in_proj_b.weight", deq(f"blk.{il}.ssm_beta.weight"))
        put(f"{LA}.in_proj_a.weight", deq(f"blk.{il}.ssm_alpha.weight"))
        conv = deq(f"blk.{il}.ssm_conv1d.weight")   # gguf: (k, conv_dim) 또는 (conv_dim, k)
        cd = tc['linear_key_head_dim']*tc['linear_num_key_heads']*2 + tc['linear_value_head_dim']*tc['linear_num_value_heads']
        if conv.shape[0] == cd: conv3 = conv[:, :, None]
        else:                   conv3 = conv.T[:, :, None]
        put(f"{LA}.conv1d.weight", conv3)
        put(f"{LA}.dt_bias", deq(f"blk.{il}.ssm_dt.bias"))
        put(f"{LA}.A_log", deq(f"blk.{il}.ssm_a"))
        put(f"{LA}.norm.weight", deq(f"blk.{il}.ssm_norm.weight"))
        put(f"{LA}.out_proj.weight", deq(f"blk.{il}.ssm_out.weight"))

    # MoE: 라우터·전문가는 물리층 wl, shexp 는 논리층 il (SHEXP_LOGICAL=1)
    M = f"{P}.mlp"
    put(f"{M}.gate.weight", deq(f"blk.{wl}.ffn_gate_inp.weight"))
    for role, mlxname in (("gate","gate_proj"), ("up","up_proj"), ("down","down_proj")):
        put(f"{M}.switch_mlp.{mlxname}.weight", deq(f"blk.{wl}.ffn_{role}_exps.weight"))
    for role, mlxname in (("gate","gate_proj"), ("up","up_proj"), ("down","down_proj")):
        put(f"{M}.shared_expert.{mlxname}.weight", deq(f"blk.{il}.ffn_{role}_shexp.weight"))
    put(f"{M}.shared_expert_gate.weight", deq(f"blk.{il}.ffn_gate_inp_shexp.weight").reshape(1, -1))
    # 핫 전문가 (있으면)
    if f"blk.{il}.ffn_hot_inp.weight" in T:
        put(f"{M}.hot_gate.weight", deq(f"blk.{il}.ffn_hot_inp.weight"))
        for role, mlxname in (("gate","gate_proj"), ("up","up_proj"), ("down","down_proj")):
            put(f"{M}.hot_mlp.{mlxname}.weight", deq(f"blk.{il}.ffn_{role}_hexps.weight"))

os.makedirs(A.out, exist_ok=True)
tot = sum(v.nbytes for v in out.values())
print(f"텐서 {len(out)}개, fp32 {tot/1e9:.2f} GB → {A.bits}bit 양자화 저장")
mx.save_safetensors(os.path.join(A.out, "model_fp32.safetensors"), out)
json.dump({"model_type": "qwen3_5_moe", "text_config": tc, "fold_map": WL,
           "quantization": {"group_size": 64, "bits": A.bits}},
          open(os.path.join(A.out, "config.json"), "w"), indent=1)
print("완료:", A.out)
