#!/usr/bin/env python3
"""접힌 Qwen3.5-122B-A10B 의 MLX 구현. llama.cpp qwen35moe.cpp build_layer_ffn 을 그대로 옮긴다.

FFN(x) = γ·MoE(rep 라우터, rep 전문가) + δ·MoE(자기 핫 라우터, 자기 핫 전문가)
         + sigmoid(자기 shexp 게이트)·SwiGLU(자기 shexp)
γ=0 인 접힌 층은 MoE 분기를 아예 만들지 않는다(항등이므로 수치 동일).

전문가는 물리층 모듈 하나만 존재하고, 접힌 층은 밑줄 속성(_rep)으로 참조한다.
밑줄 이름은 MLX 파라미터 순회에서 제외되므로 중복 등록/중복 기울기가 없다.
"""
import json, os, glob
from typing import Optional, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3_5 import TextModelArgs, GatedDeltaNet, create_ssm_mask
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_next import Qwen3NextAttention as Attention
from mlx_lm.models.qwen3_next import Qwen3NextMLP as MLP
from mlx_lm.models.switch_layers import SwitchGLU


def route(gates: mx.array, k: int):
    """llama.cpp build_moe_ffn(SOFTMAX, norm_w=true): softmax → top-k → 재정규화."""
    g = mx.softmax(gates, axis=-1, precise=True)
    inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(g, inds, axis=-1)
    return inds, scores / scores.sum(axis=-1, keepdims=True)


class FoldGDN(GatedDeltaNet):
    """llama.cpp 의 융합 GDN 은 q/k 헤드를 **타일**(v헤드 j → k헤드 j % Hk)로 확장한다.
    MLX 기본은 블록(j // R). 실측(heal/s5.log, 층 0):
        타일 + q=l2/√d → rel 3.29e-02   (다른 텐서의 이중양자화 오차와 동급)
        블록(MLX 기본)  → rel 1.06e+01
    파라미터 이름은 그대로이므로 가중치 로드에는 영향이 없다."""

    def __call__(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [t.reshape(B, S, hh, dd) for t, hh, dd in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim])]

        inv = self.head_k_dim ** -0.5
        q = (inv * inv) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv * mx.fast.rms_norm(k, None, 1e-6)
        r = self.num_v_heads // self.num_k_heads
        if r > 1:
            q = mx.concatenate([q] * r, axis=2)     # 타일: j → j % Hk
            k = mx.concatenate([k] * r, axis=2)

        state = cache[1] if cache else None
        out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias,
                                        state, mask, use_kernel=not self.training)
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))


class RepMoe(nn.Module):
    """대표/고정 물리층: 자기 라우터·전문가 + 자기 공유전문가."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def moe_only(self, x):
        inds, scores = route(self.gate(x), self.top_k)
        return (self.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)

    def shexp(self, x):
        return mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    def __call__(self, x):
        return self.moe_only(x) + self.shexp(x)


class FoldedMoe(nn.Module):
    """접힌 층: γ·(물리층 MoE) + δ·(자기 핫 MoE) + 자기 공유전문가."""

    def __init__(self, args: TextModelArgs, gamma: float, delta: float, n_hot: int = 0):
        super().__init__()
        self.gamma, self.delta, self.top_k = gamma, delta, args.num_experts_per_tok
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)
        if n_hot:
            # llama.cpp: 사용 전문가 수는 16 캡 (Metal mul_mat_id 절벽)
            self.n_hot_used = min(n_hot, 16)
            self.hot_gate = nn.Linear(args.hidden_size, n_hot, bias=False)
            self.hot_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, n_hot)
        self._rep = None      # 밑줄: 파라미터 순회 제외 (물리층 모듈 참조)

    def shexp(self, x):
        return mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    def __call__(self, x):
        out = self.shexp(x)
        if self.gamma != 0.0:
            out = out + self.gamma * self._rep.moe_only(x)
        if self.delta != 0.0 and "hot_gate" in self:
            inds, scores = route(self.hot_gate(x), self.n_hot_used)
            hot = (self.hot_mlp(x, inds) * scores[..., None]).sum(axis=-2)
            out = out + self.delta * hot
        return out


class Layer(nn.Module):
    def __init__(self, args: TextModelArgs, il: int, mlp: nn.Module):
        super().__init__()
        self.is_linear = (il + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = FoldGDN(args)
        else:
            self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = mlp
        self._ckpt = None      # 밑줄: 파라미터 순회 제외 (nn.utils.checkpoint 래퍼)

    def __call__(self, x, mask=None, cache=None):
        attn = self.linear_attn if self.is_linear else self.self_attn
        h = x + attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class FoldedModel(nn.Module):
    def __init__(self, args: TextModelArgs, fold_map, gamma, hot_layers, delta=1.0, n_hot=16):
        super().__init__()
        self.args, self.fold_map = args, fold_map
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        mlps = []
        for il, wl in enumerate(fold_map):
            if wl == il:
                mlps.append(RepMoe(args))
            else:
                mlps.append(FoldedMoe(args, gamma[il], delta,
                                      n_hot if (il in hot_layers and gamma[il] != 0.0) else 0))
        self.layers = [Layer(args, il, mlps[il]) for il in range(len(fold_map))]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.fa_idx = args.full_attention_interval - 1

    def wire(self):
        """접힌 층 → 물리층 전문가 결선. 가중치 로드 후 한 번 호출."""
        for il, wl in enumerate(self.fold_map):
            if wl != il:
                self.layers[il].mlp._rep = self.layers[wl].mlp

    def __call__(self, inputs, cache=None):
        h = self.embed(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            fn = layer._ckpt if layer._ckpt is not None else layer
            h = fn(h, (ssm_mask if layer.is_linear else fa_mask), c)
        return self.lm_head(self.norm(h))

    @property
    def layers_for_cache(self):
        return self.layers


def load(path="heal/mlx-fold", delta=None):
    cfg = json.load(open(f"{path}/fold.json"))
    tc = cfg["text_config"]
    args = TextModelArgs.from_dict({**tc, "model_type": "qwen3_5_moe"}) \
        if "text_config" not in tc else TextModelArgs.from_dict(tc)
    model = FoldedModel(args, cfg["fold_map"], cfg["gamma"], set(cfg["hot_layers"]),
                        delta if delta is not None else cfg["delta"])
    # 저장된 비트폭대로 양자화 (샤드 키에 scales 가 있는 모듈만)
    qmap = cfg["manifest"].get("quantized", {})
    gs = cfg["group_size"]
    nn.quantize(model, group_size=gs, bits=cfg["expert_bits"],
                class_predicate=lambda p, m: p in qmap and qmap[p] == cfg["expert_bits"])
    nn.quantize(model, group_size=gs, bits=cfg["attn_bits"],
                class_predicate=lambda p, m: p in qmap and qmap[p] == cfg["attn_bits"])
    w = {}
    for f in sorted(glob.glob(f"{path}/shard-*.safetensors")):
        w.update(mx.load(f))
    model.load_weights(list(w.items()), strict=True)
    model.wire()
    mx.eval(model.parameters())
    return model, args, cfg
