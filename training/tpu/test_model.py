#!/usr/bin/env python3
"""python test_model.py — CPU에서 몇 초. TPU 세션 태우기 전에 여기서 걸러낸다."""
import math

import numpy as np
import jax
import jax.numpy as jnp
import optax

from model import CONFIGS, TOKENS, GPT, GPTConfig, init_params, param_count
from train import lr_at, WARMUP, LR, LR_MIN, train_batch

TINY = GPTConfig(vocab_size=64, ctx=16, d=32, n_head=2, ffn_hidden=88,
                 n_pre=1, n_loop_layers=2, n_loops=3, n_post=1)


def test_param_counts_match_formula():
    for name, cfg in CONFIGS.items():
        _, p = init_params(cfg)
        assert param_count(p) == int(cfg.n_params), (name, param_count(p), cfg.n_params)


def test_loop_shares_weights():
    """loop arm은 루프 임베딩 R·d 만큼만 커야 한다. 그보다 크면 공유가 깨진 것."""
    for s in ("15m", "50m", "150m"):
        small, loop = CONFIGS[f"{s}-small"], CONFIGS[f"{s}-loop"]
        _, p = init_params(loop)
        extra = param_count(p) - small.n_params
        assert extra == loop.n_loops * loop.d, (s, extra)


def test_ladder_shape():
    n = [int(CONFIGS[f"{s}-small"].n_params) for s in ("15m", "50m", "150m")]
    assert n == [15_208_960, 49_590_720, 143_358_592], n
    assert TOKENS["150m"] == 13 * n[2]
    # large는 2× depth → 파라미터는 1.8~2.0×
    for s in ("15m", "50m", "150m"):
        r = CONFIGS[f"{s}-large"].n_params / CONFIGS[f"{s}-small"].n_params
        assert 1.8 < r < 2.0, (s, r)


def test_causal():
    """뒤 토큰을 바꿔도 앞 위치 logit이 흔들리면 마스킹이 샌 것."""
    model, p = init_params(TINY)
    x = jnp.array(np.random.default_rng(0).integers(0, 64, (1, 16)))
    y = x.at[0, 10].set((x[0, 10] + 1) % 64)
    a, b = model.apply({"params": p}, x), model.apply({"params": p}, y)
    assert jnp.allclose(a[:, :10], b[:, :10], atol=2e-2)
    assert not jnp.allclose(a[:, 10:], b[:, 10:], atol=2e-2)


def test_lr_schedule_matches_torch():
    ms = 6103
    for step in (0, 1, 299, 300, 3000, 6102):
        want = (LR * (step + 1) / WARMUP if step < WARMUP else
                LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * min(1.0, step / ms))))
        assert abs(float(lr_at(step, ms)) - want) < 1e-9, step


def test_train_batch_is_resumable_and_shifted():
    arr = np.arange(10_000, dtype=np.uint16)
    x, y = train_batch(arr, 42, 1337, 2, 4, 8)
    assert x.shape == y.shape == (2, 4, 8)
    assert (y[..., :-1] == x[..., 1:]).all()          # y는 x를 한 칸 민 것
    x2, _ = train_batch(arr, 42, 1337, 2, 4, 8)
    assert (x == x2).all()                            # 재개해도 같은 배치
    x3, _ = train_batch(arr, 43, 1337, 2, 4, 8)
    assert not (x == x3).all()


def test_overfits_one_batch():
    """실제로 학습이 되는지. ln(64)=4.16에서 유의미하게 내려가야 한다."""
    model, p = init_params(TINY)
    tx = optax.adamw(3e-3)
    st = tx.init(p)
    rng = np.random.default_rng(0)
    x = jnp.array(rng.integers(0, 64, (4, 16)))
    y = jnp.array(rng.integers(0, 64, (4, 16)))

    def loss_fn(p):
        return optax.softmax_cross_entropy_with_integer_labels(
            model.apply({"params": p}, x), y).mean()

    first = float(loss_fn(p))
    step = jax.jit(lambda p, st: (lambda l, g: (l, *tx.update(g, st, p)))(*jax.value_and_grad(loss_fn)(p)))
    for _ in range(60):
        l, upd, st = step(p, st)
        p = optax.apply_updates(p, upd)
    last = float(loss_fn(p))
    assert first > 4.0, first
    assert last < 1.0, (first, last)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
