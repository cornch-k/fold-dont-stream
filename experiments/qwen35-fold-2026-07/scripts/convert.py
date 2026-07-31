#!/usr/bin/env python3
"""접힌 GGUF(v2a-hot) → MLX 샤드 가중치. 텐서 하나씩 역양자화→재양자화해 스트리밍한다.

비트폭은 원본 GGUF 를 따라간다: 전문가/핫 q3_K→3bit(3.5bpw), 어텐션 q4_K→4bit,
공유전문가 q6_K→bf16(학습 대상이라 양자화하지 않음), 라우터·norm→fp32.

접기 결선: 논리층 il 의 FFN 전문가/라우터/FFN pre-norm 은 물리층 wl 것을 쓴다.
따라서 전문가는 물리층에만 저장하고, 접힌 층은 shexp/핫/어텐션만 저장한다.
"""
import sys, os, json, argparse, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanfix', 'gguf-py'))
import numpy as np
import mlx.core as mx
from gguf import GGUFReader
import gguf.quants as Q

AP = argparse.ArgumentParser()
AP.add_argument("--gguf", default="models/qwen35-v2a-hot.gguf")
AP.add_argument("--out", default="heal/mlx-fold")
AP.add_argument("--reps", type=int, default=6)
AP.add_argument("--fixfrom", type=int, default=44)
AP.add_argument("--gamma-file", default="exp/g46.txt")
AP.add_argument("--expert-bits", type=int, default=3)
AP.add_argument("--attn-bits", type=int, default=4)
AP.add_argument("--group", type=int, default=64)
A = AP.parse_args()

NL = 48   # 논리층 (blk.48 = MTP, 제외)
GS = A.group


def build_map(G, F, L=NL):
    return [il if il >= F else ((il * G // F) * F + G - 1) // G for il in range(L)]


WL = build_map(A.reps, A.fixfrom)
PHYS = sorted(set(WL))
GAMMA = [float(x) for x in open(A.gamma_file)][:NL]
# γ 는 접힌 층에만 걸린다 (qwen35moe.cpp: sl != wl 조건). 대표층 값은 무의미.
GAM = [0.0 if WL[il] == il else GAMMA[il] for il in range(NL)]
print(f"물리층 {len(PHYS)}개 {PHYS}")
print(f"MoE 생략 층(γ=0, 접힘): {[il for il in range(NL) if WL[il] != il and GAM[il] == 0.0]}")

r = GGUFReader(A.gguf)
T = {t.name: t for t in r.tensors}
HOT = sorted(int(k.split('.')[1]) for k in T if k.endswith('.ffn_hot_inp.weight'))
print(f"핫 텐서 보유 층: {HOT}")


def deq(name):
    t = T[name]
    if t.tensor_type == 0:
        return np.asarray(t.data, dtype=np.float32)
    return Q.dequantize(t.data, t.tensor_type).astype(np.float32)


os.makedirs(A.out, exist_ok=True)
manifest = {}


def emit(shard, out, key, name, *, bits=None, raw=None, transform=None):
    """GGUF 텐서 하나를 MLX 파라미터로. bits=None 이면 bf16/fp32 그대로."""
    a = raw if raw is not None else deq(name)
    if transform is not None:
        a = transform(a)
    x = mx.array(a)
    del a
    if bits is None:
        out[key] = x.astype(mx.float32 if x.ndim <= 1 else mx.bfloat16)
    else:
        w, s, b = mx.quantize(x.astype(mx.bfloat16), group_size=GS, bits=bits)
        out[key], out[key[:-7] + ".scales"], out[key[:-7] + ".biases"] = w, s, b
        manifest.setdefault("quantized", {})[key[:-7]] = bits
    del x
    gc.collect()


EB, AB = A.expert_bits, A.attn_bits

# ── 샤드 0: 임베딩/출력 ──
out = {}
emit(0, out, "embed.weight", "token_embd.weight", bits=AB)
emit(0, out, "norm.weight", "output_norm.weight")
emit(0, out, "lm_head.weight", "output.weight", bits=AB)
mx.save_safetensors(f"{A.out}/shard-misc.safetensors", out)
print(f"shard-misc: {len(out)}개")
del out
gc.collect()
mx.clear_cache()

# ── 층별 샤드 ──
for il in range(NL):
    wl, gam = WL[il], GAM[il]
    is_rep = (wl == il)
    is_full = ((il + 1) % 4 == 0)
    P = f"layers.{il}"
    out = {}
    emit(il, out, f"{P}.input_layernorm.weight", f"blk.{il}.attn_norm.weight")
    # FFN pre-norm 은 물리층(norm_phys=true 기본)
    emit(il, out, f"{P}.post_attention_layernorm.weight", f"blk.{wl}.post_attention_norm.weight")

    if is_full:
        S = f"{P}.self_attn"
        emit(il, out, f"{S}.q_proj.weight", f"blk.{il}.attn_q.weight", bits=AB)
        emit(il, out, f"{S}.k_proj.weight", f"blk.{il}.attn_k.weight", bits=AB)
        emit(il, out, f"{S}.v_proj.weight", f"blk.{il}.attn_v.weight", bits=AB)
        emit(il, out, f"{S}.o_proj.weight", f"blk.{il}.attn_output.weight", bits=AB)
        emit(il, out, f"{S}.q_norm.weight", f"blk.{il}.attn_q_norm.weight")
        emit(il, out, f"{S}.k_norm.weight", f"blk.{il}.attn_k_norm.weight")
    else:
        S = f"{P}.linear_attn"
        emit(il, out, f"{S}.in_proj_qkv.weight", f"blk.{il}.attn_qkv.weight", bits=AB)
        emit(il, out, f"{S}.in_proj_z.weight", f"blk.{il}.attn_gate.weight", bits=AB)
        emit(il, out, f"{S}.out_proj.weight", f"blk.{il}.ssm_out.weight", bits=AB)
        emit(il, out, f"{S}.in_proj_b.weight", f"blk.{il}.ssm_beta.weight")
        emit(il, out, f"{S}.in_proj_a.weight", f"blk.{il}.ssm_alpha.weight")
        emit(il, out, f"{S}.dt_bias", f"blk.{il}.ssm_dt.bias")
        # llama.cpp 는 ssm_a 에 -exp(A_log) 를 저장한다 (qwen35moe.cpp: "-A_log.exp() * softplus").
        # MLX gated_delta_update 는 A_log 를 받아 내부에서 -exp 를 취하므로 되돌려야 한다.
        emit(il, out, f"{S}.A_log", f"blk.{il}.ssm_a", transform=lambda a: np.log(-a))
        emit(il, out, f"{S}.norm.weight", f"blk.{il}.ssm_norm.weight")
        # gguf (conv_dim, k) → mlx Conv1d (out_ch, k, in_ch/groups=1)
        emit(il, out, f"{S}.conv1d.weight", f"blk.{il}.ssm_conv1d.weight",
             transform=lambda a: a[:, :, None])

    M = f"{P}.mlp"
    # 공유 전문가는 논리층 (SHEXP_LOGICAL=1) — 학습 대상이라 bf16 유지
    for role, mlxname in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
        emit(il, out, f"{M}.shared_expert.{mlxname}.weight", f"blk.{il}.ffn_{role}_shexp.weight")
    emit(il, out, f"{M}.shared_expert_gate.weight", f"blk.{il}.ffn_gate_inp_shexp.weight",
         transform=lambda a: a.reshape(1, -1))

    # 라우터/전문가는 물리층에만 저장
    if is_rep:
        emit(il, out, f"{M}.gate.weight", f"blk.{il}.ffn_gate_inp.weight")
        for role, mlxname in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            emit(il, out, f"{M}.switch_mlp.{mlxname}.weight", f"blk.{il}.ffn_{role}_exps.weight", bits=EB)

    # 핫 전문가: 접힌 층 + 텐서 존재 + γ 분기가 살아있는 층만 그래프에 오른다
    if not is_rep and il in HOT:
        emit(il, out, f"{M}.hot_gate.weight", f"blk.{il}.ffn_hot_inp.weight")
        for role, mlxname in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            emit(il, out, f"{M}.hot_mlp.{mlxname}.weight", f"blk.{il}.ffn_{role}_hexps.weight", bits=EB)

    mx.save_safetensors(f"{A.out}/shard-{il:02d}.safetensors", out)
    nb = sum(v.nbytes for v in out.values())
    print(f"shard-{il:02d}: {len(out):2d}개 {nb/1e6:7.1f} MB  wl={wl} γ={gam} rep={is_rep} hot={il in HOT}", flush=True)
    del out
    gc.collect()
    mx.clear_cache()

cfg = json.load(open('exp/qwen35-config.json'))
tc = cfg.get('text_config', cfg)
tc = dict(tc)
tc['rope_theta'] = 1e7          # GGUF qwen35moe.rope.freq_base (config.json 과 다름 — GGUF 우선)
tc['rms_norm_eps'] = 1e-6
tc['vocab_size'] = 248320
json.dump({"text_config": tc, "fold_map": WL, "gamma": GAM, "hot_layers": HOT,
           "phys": PHYS, "group_size": GS, "expert_bits": EB, "attn_bits": AB,
           "delta": 1.0, "manifest": manifest},
          open(f"{A.out}/fold.json", "w"), indent=1)
tot = sum(os.path.getsize(f"{A.out}/{f}") for f in os.listdir(A.out) if f.endswith('.safetensors'))
print(f"완료: {A.out}  총 {tot/1e9:.2f} GB")
