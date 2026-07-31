#!/usr/bin/env python3
"""힐링 어댑터를 GGUF 로 되돌린다. 추론 비용 증가 0 — 전부 원래 있던 텐서 자리다.

  · 논리층별 attn pre-norm  (blk.N.attn_norm.weight)          F32 그대로 교체
  · 논리층별 FFN pre-norm   (blk.N.post_attention_norm.weight) F32, **접힌 층에도 기록**
      → 실행 시 LLAMA_FOLD_NORM_LOGICAL=1 로 논리층 norm 을 쓰게 한다
  · 논리층별 공유전문가      (blk.N.ffn_{gate,up,down}_shexp)   LoRA 병합 후 Q6_K 재양자화
  · 논리층별 공유전문가 게이트(blk.N.ffn_gate_inp_shexp)        F32
  · 접힌 층 핫 라우터        (blk.N.ffn_hot_inp)                F32

전문가 본체·어텐션은 학습하지 않았으므로 원본 바이트를 그대로 복사한다.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanfix', 'gguf-py'))
import numpy as np
import mlx.core as mx
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGMLQuantizationType
import gguf.quants as Q

AP = argparse.ArgumentParser()
AP.add_argument("--gguf", default="models/qwen35-v2a-hot.gguf")
AP.add_argument("--out", default="models/qwen35-v2a-healed.gguf")
AP.add_argument("--adapter", default="heal/adapter/adapter.safetensors")
AP.add_argument("--fold", default="heal/mlx-fold/fold.json")
A = AP.parse_args()

cfg = json.load(open(A.fold))
acfg = json.load(open(os.path.join(os.path.dirname(A.adapter), "adapter_config.json")))
SCALE, R = acfg["scale"], acfg["r"]
W = {k: v for k, v in mx.load(A.adapter).items()}
print(f"어댑터 {len(W)}개 텐서, LoRA r={R} scale={SCALE}")


def f32(k):
    return np.array(W[k].astype(mx.float32)) if k in W else None


def lora_delta(prefix):
    """LoRALinear.fuse 와 동일: delta = (scale · lora_b.T) @ lora_a.T"""
    a, b = f32(f"{prefix}.lora_a"), f32(f"{prefix}.lora_b")
    if a is None or b is None:
        return None
    return (SCALE * b.T) @ a.T


r = GGUFReader(A.gguf)
w = GGUFWriter(A.out, "qwen35moe")
for field in r.fields.values():
    if field.name in ("GGUF.tensor_count", "GGUF.kv_count", "general.architecture"):
        continue
    try:
        w.add_key_value(field.name, field.contents(), field.types[0])
    except Exception as e:
        print(f"  kv 건너뜀 {field.name}: {e}")

SHEXP = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
n_rep = 0
for t in r.tensors:
    name = t.name
    new = None
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "blk":
        il = int(parts[1])
        tail = ".".join(parts[2:])
        P = f"layers.{il}"
        if tail == "attn_norm.weight":
            new = f32(f"{P}.input_layernorm.weight")
        elif tail == "post_attention_norm.weight":
            new = f32(f"{P}.post_attention_layernorm.weight")
        elif tail == "ffn_gate_inp_shexp.weight":
            v = f32(f"{P}.mlp.shared_expert_gate.weight")
            new = None if v is None else v.reshape(-1)
        elif tail == "ffn_hot_inp.weight":
            new = f32(f"{P}.mlp.hot_gate.weight")
        elif tail.startswith("ffn_") and tail.endswith("_shexp.weight"):
            role = tail[len("ffn_"):-len("_shexp.weight")]
            d = lora_delta(f"{P}.mlp.shared_expert.{SHEXP[role]}.linear")
            if d is not None:
                base = Q.dequantize(t.data, t.tensor_type).astype(np.float32)
                new = base + d.astype(np.float32)

    if new is None:
        w.add_tensor(name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
    else:
        n_rep += 1
        if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            w.add_tensor(name, new.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
        else:   # shexp: 원래 비트폭(Q6_K)으로 재양자화
            q = Q.quantize(new, t.tensor_type)
            w.add_tensor(name, q, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
        print(f"  교체 {name}", flush=True)

print(f"교체 {n_rep}개 / 전체 {len(r.tensors)}개 → {A.out}")
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("병합 완료")
