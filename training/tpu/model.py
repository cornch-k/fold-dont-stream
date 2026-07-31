"""순환 깊이 GPT (JAX/Flax). training/rdepth/model.py의 dense 경로 포트.

pre → (loop block × R) → post, middle-cycle 공유. Phase 1은 dense 전용 —
MoE는 Phase 1 게이트 통과 후에 추가한다.

파라미터 수 공식 (tied embedding, bias 없음, head_dim=64, ffn=2.75d):
    B      = n_pre + n_loop_layers + n_post
    block  = 4d² (attn qkv+proj) + 3·2.75d² (SwiGLU) = 12.25d²
    total  = 12.25·B·d² + (vocab + ctx + 2B+1 + R)·d      # 2B+1 = RMSNorm scale
loop arm은 n_loops만 늘리므로 small arm과 블록 파라미터가 정확히 같다.
루프 임베딩 R·d 만큼만 더 크다 (15M에서 1,536개, 0.01%).
"""
import dataclasses
import functools

import jax
import jax.numpy as jnp
import flax.linen as nn

COMPUTE_DTYPE = jnp.bfloat16   # 행렬곱만 bf16, residual/norm/loss는 fp32
PARAM_DTYPE = jnp.float32
INIT_STD = 0.02


@dataclasses.dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 4096
    ctx: int = 512
    d: int = 512
    n_head: int = 8
    ffn_hidden: int = 1408
    n_pre: int = 1
    n_loop_layers: int = 2
    n_loops: int = 1
    n_post: int = 1
    inject_input: bool = True
    use_loop_emb: bool = True

    @property
    def n_params(self):
        b = self.n_pre + self.n_loop_layers + self.n_post
        loop_emb = self.n_loops if self.use_loop_emb and self.n_loops > 1 else 0
        return int(12.25 * self.d ** 2 * b
                   + (self.vocab_size + self.ctx + 2 * b + 1 + loop_emb) * self.d)


def _dense(features, name):
    return nn.Dense(features, use_bias=False, name=name, dtype=COMPUTE_DTYPE,
                    param_dtype=PARAM_DTYPE, kernel_init=nn.initializers.normal(INIT_STD))


def _norm(name):
    # norm은 fp32 유지: bf16 RMS는 d=1664에서 눈에 띄게 흔들린다
    return nn.RMSNorm(epsilon=1e-6, name=name, dtype=jnp.float32, param_dtype=PARAM_DTYPE)


def _embed(num, d, name):
    return nn.Embed(num, d, name=name, dtype=jnp.float32, param_dtype=PARAM_DTYPE,
                    embedding_init=nn.initializers.normal(INIT_STD))


class Block(nn.Module):
    cfg: GPTConfig

    @nn.compact
    def __call__(self, x):
        c = self.cfg
        B, T, _ = x.shape
        heads = (B, T, c.n_head, c.d // c.n_head)
        h = _norm("ln1")(x)
        q, k, v = jnp.split(_dense(3 * c.d, "qkv")(h), 3, axis=-1)
        a = jax.nn.dot_product_attention(q.reshape(heads), k.reshape(heads),
                                         v.reshape(heads), is_causal=True)
        x = x + _dense(c.d, "proj")(a.reshape(B, T, c.d))
        h = _norm("ln2")(x)
        x = x + _dense(c.d, "w2")(jax.nn.silu(_dense(c.ffn_hidden, "w1")(h))
                                  * _dense(c.ffn_hidden, "w3")(h))
        return x


class GPT(nn.Module):
    cfg: GPTConfig

    def setup(self):
        c = self.cfg
        self.tok_emb = _embed(c.vocab_size, c.d, "tok_emb")
        self.pos_emb = _embed(c.ctx, c.d, "pos_emb")
        self.pre = [Block(c, name=f"pre_{i}") for i in range(c.n_pre)]
        self.loop = [Block(c, name=f"loop_{i}") for i in range(c.n_loop_layers)]
        self.post = [Block(c, name=f"post_{i}") for i in range(c.n_post)]
        self.loop_emb = (_embed(c.n_loops, c.d, "loop_emb")
                         if c.use_loop_emb and c.n_loops > 1 else None)
        self.ln_f = _norm("ln_f")

    def __call__(self, idx):
        c = self.cfg
        x = self.tok_emb(idx) + self.pos_emb(jnp.arange(idx.shape[1]))
        h0 = x
        for b in self.pre:
            x = b(x)
        for r in range(c.n_loops):
            if c.inject_input and c.n_loops > 1:
                x = x + h0
            if self.loop_emb is not None:
                x = x + self.loop_emb.embedding[r]
            for b in self.loop:            # 같은 인스턴스 재호출 = 파라미터 공유
                x = b(x)
        for b in self.post:
            x = b(x)
        x = self.ln_f(x)
        return jnp.einsum("btd,vd->btv", x.astype(COMPUTE_DTYPE),
                          self.tok_emb.embedding.astype(COMPUTE_DTYPE)).astype(jnp.float32)


def _cfg(d, **kw):
    return GPTConfig(d=d, n_head=d // 64, ffn_hidden=int(d * 2.75), **kw)


# 사다리: d만 스케일. 구조(1 pre / 2 loop / 1 post)와 ctx·vocab은 고정 —
# 회복률을 스케일 하나의 축에서만 움직이게 하려는 것.
SCALES = {"15m": 512, "50m": 960, "150m": 1664}
ARMS = {
    "small": dict(n_loop_layers=2, n_loops=1),   # 기준
    "loop":  dict(n_loop_layers=2, n_loops=3),   # small과 동일 파라미터, R=3
    "large": dict(n_loop_layers=6, n_loops=1),   # 2× depth (≈1.9× 파라미터)
}
CONFIGS = {f"{s}-{a}": _cfg(d, **kw) for s, d in SCALES.items() for a, kw in ARMS.items()}

# 토큰 예산은 스케일당 고정 = 13 × (small arm 파라미터). 세 arm이 같은 토큰을
# 보아야 회복률이 성립하므로, large가 더 크다고 토큰을 늘리지 않는다.
TOKENS = {s: int(13 * CONFIGS[f"{s}-small"].n_params) for s in SCALES}


def param_count(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def init_params(cfg, seed=0):
    model = GPT(cfg)
    idx = jnp.zeros((1, cfg.ctx), jnp.int32)
    return model, model.init(jax.random.PRNGKey(seed), idx)["params"]
