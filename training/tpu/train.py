#!/usr/bin/env python3
"""Phase 1 dense 학습 (JAX/Flax, TPU). training/rdepth/train.py와 동일 하이퍼.

15M 게이트 런은 PyTorch 참조와 같은 조건이어야 하므로 배치·LR·워밍업·시드를
바꾸지 않는다. 스케일 간에도 고정 — arm 비교는 스케일 안에서만 하므로
LR이 스케일에 최적이 아닌 것은 회복률을 편향시키지 않는다.

  python train.py --run 15m-loop --data-dir data/tinystories
  python train.py --run 150m-small --data-dir data/fineweb --accum 2 --resume
  python train.py --run 15m-small --data-dir data/tinystories --max-seconds 600  # CU 측정
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from jax.sharding import NamedSharding, PartitionSpec as P

from model import CONFIGS, TOKENS, GPT, param_count

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, EFF_BATCH_TOK, WARMUP, LR, LR_MIN, WD, CLIP = 1337, 32768, 300, 6e-4, 6e-5, 0.1, 1.0
EVAL_EVERY, EVAL_BATCHES, CKPT_EVERY = 500, 100, 200
VAL_B = 64   # accum과 무관하게 고정 — 세 arm이 val.bin의 같은 구간을 본다


def lr_at(step, max_steps):
    """torch 런과 동일: 300스텝 선형 워밍업 → step/max_steps 기준 코사인."""
    warm = LR * (step + 1) / WARMUP
    cos = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + jnp.cos(math.pi * jnp.minimum(1.0, step / max_steps)))
    return jnp.where(step < WARMUP, warm, cos)


def train_batch(arr, step, seed, accum, bm, T):
    rng = np.random.default_rng([seed, step])          # 재개해도 같은 배치가 나온다
    ix = rng.integers(0, len(arr) - T - 1, size=accum * bm)
    x = np.stack([arr[i:i + T] for i in ix]).astype(np.int32)
    y = np.stack([arr[i + 1:i + 1 + T] for i in ix]).astype(np.int32)
    return x.reshape(accum, bm, T), y.reshape(accum, bm, T)


def val_batch(arr, k, B, T):
    s = k * B * T
    if s + B * T + 1 > len(arr):
        return None
    x = np.stack([arr[s + j * T: s + (j + 1) * T] for j in range(B)]).astype(np.int32)
    y = np.stack([arr[s + j * T + 1: s + (j + 1) * T + 1] for j in range(B)]).astype(np.int32)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(CONFIGS))
    ap.add_argument("--data-dir", required=True, help="train.bin / val.bin 이 있는 디렉터리")
    ap.add_argument("--tokens", type=int, default=0, help="0이면 13:1 기본값")
    ap.add_argument("--accum", type=int, default=1, help="HBM 부족할 때만. 재개 시 동일해야 함")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default=HERE, help="ckpt/ logs/ results/ 상위 (Drive 권장)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-seconds", type=int, default=0, help="CU 소모율 측정용 조기 종료")
    ap.add_argument("--eval-batches", type=int, default=EVAL_BATCHES, help="스모크 테스트용")
    ap.add_argument("--ckpt-every", type=int, default=CKPT_EVERY,
                    help="150m-large 체크포인트는 3.3GB — Drive 쓰기가 학습보다 오래 걸리면 늘릴 것")
    # Drive FUSE는 쓴 만큼 로컬 디스크에 캐시를 쌓아 디스크를 채운다.
    # 체크포인트는 로컬(/content)에, 작은 logs/results만 Drive에 두는 것이 안전하다.
    ap.add_argument("--ckpt-dir", default="", help="비우면 out-dir/ckpt")
    a = ap.parse_args()

    cfg = CONFIGS[a.run]
    scale, arm = a.run.split("-")
    corpus = os.path.basename(os.path.normpath(a.data_dir))
    tag = f"{a.run}-{corpus}" + (f"-s{a.seed}" if a.seed != SEED else "")
    total_tokens = a.tokens or TOKENS[scale]
    max_steps = total_tokens // EFF_BATCH_TOK
    T = cfg.ctx
    bm = EFF_BATCH_TOK // (T * a.accum)
    assert EFF_BATCH_TOK == bm * T * a.accum, "accum이 배치를 나누지 못함"

    devs = jax.device_count()
    assert bm % devs == 0, f"micro-batch {bm} 이 디바이스 {devs}개로 안 나뉨"
    # Auto: 배치 축만 지정하고 나머지 sharding은 XLA가 전파. Explicit(0.11 기본)로
    # 두면 임베딩 gather에서 출력 sharding을 못 정해 죽는다.
    mesh = jax.make_mesh((devs,), ("data",), axis_types=(jax.sharding.AxisType.Auto,))
    repl = NamedSharding(mesh, P())
    dsh = NamedSharding(mesh, P(None, "data", None))

    model = GPT(cfg)
    params = jax.jit(lambda k: model.init(k, jnp.zeros((1, T), jnp.int32))["params"],
                     out_shardings=repl)(jax.random.PRNGKey(a.seed))
    n_params = param_count(params)
    tx = optax.chain(optax.clip_by_global_norm(CLIP),
                     optax.adamw(lambda s: lr_at(s, max_steps), b1=0.9, b2=0.95, weight_decay=WD))
    opt_state = jax.jit(tx.init, out_shardings=repl)(params)

    print(f"run={a.run} corpus={corpus} params={n_params:,} (공식 {cfg.n_params:,.0f}) "
          f"tokens={total_tokens:,} steps={max_steps} micro={bm}x{T}x{a.accum} "
          f"devices={devs} {jax.devices()[0].device_kind}", flush=True)

    def loss_fn(p, x, y):
        logits = model.apply({"params": p}, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def train_step(params, opt_state, xs, ys):
        def body(carry, xy):
            g_acc, l_acc = carry
            l, g = jax.value_and_grad(loss_fn)(params, *xy)
            return (jax.tree.map(jnp.add, g_acc, g), l_acc + l), None
        init = (jax.tree.map(jnp.zeros_like, params), jnp.float32(0.0))
        (grads, loss), _ = jax.lax.scan(body, init, (xs, ys))
        n = xs.shape[0]
        updates, opt_state = tx.update(jax.tree.map(lambda g: g / n, grads), opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss / n

    eval_step = jax.jit(loss_fn)

    for sub in ("logs", "results"):
        os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)
    ckpt_dir = a.ckpt_dir or os.path.join(a.out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.msgpack")
    log_path = os.path.join(a.out_dir, "logs", f"{tag}.csv")
    res_path = os.path.join(a.out_dir, "results", f"{tag}.json")

    step = 0
    if a.resume and os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            st = serialization.from_bytes({"params": params, "opt": opt_state, "step": 0}, f.read())
        params, opt_state, step = st["params"], st["opt"], int(st["step"])
        params, opt_state = jax.device_put((params, opt_state), repl)
        print(f"resumed at step {step}", flush=True)

    def save():
        tmp = ckpt_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(serialization.to_bytes({"params": jax.device_get(params),
                                            "opt": jax.device_get(opt_state), "step": step}))
        os.replace(tmp, ckpt_path)   # Drive에서도 부분 쓰기가 남지 않게

    train_arr = np.memmap(os.path.join(a.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_arr = np.memmap(os.path.join(a.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    assert len(train_arr) >= total_tokens, (
        f"train.bin {len(train_arr):,} 토큰 < 요구 {total_tokens:,} — 반복 학습이 됩니다")

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "tokens", "train_loss", "val_loss", "tok_s", "elapsed_s"])

    def validate():
        ls = []
        for k in range(a.eval_batches):
            b = val_batch(val_arr, k, VAL_B, T)
            if b is None:
                break
            ls.append(float(eval_step(params, *jax.device_put(b, NamedSharding(mesh, P("data", None))))))
        return sum(ls) / len(ls)

    t0, tok0, step0, vl = time.time(), step * EFF_BATCH_TOK, step, float("nan")
    while step < max_steps:
        xs, ys = train_batch(train_arr, step, a.seed, a.accum, bm, T)
        params, opt_state, loss = train_step(params, opt_state, *jax.device_put((xs, ys), dsh))
        step += 1
        if step % 50 == 0:
            el, done = time.time() - t0, step * EFF_BATCH_TOK - tok0
            print(f"step {step}/{max_steps} loss {float(loss):.4f} {done/el:,.0f} tok/s "
                  f"eta {(max_steps-step)*el/max(1,step-step0)/60:.1f}m", flush=True)
        if step % a.ckpt_every == 0:
            save()
        if step % EVAL_EVERY == 0 or step == max_steps:
            vl, el = validate(), time.time() - t0
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, step * EFF_BATCH_TOK, f"{float(loss):.4f}",
                                        f"{vl:.4f}", f"{(step*EFF_BATCH_TOK-tok0)/el:.0f}", f"{el:.0f}"])
            save()
            print(f"  [eval] step {step} val {vl:.4f}", flush=True)
        if a.max_seconds and time.time() - t0 > a.max_seconds:
            el = time.time() - t0
            rate = (step * EFF_BATCH_TOK - tok0) / el
            print(f"\n[측정] {el:.0f}s 동안 {step} 스텝, {rate:,.0f} tok/s\n"
                  f"[측정] 이 런 전체 예상 {total_tokens/rate/3600:.2f} 시간", flush=True)
            save()
            return

    if math.isnan(vl):   # 끝난 런을 --resume 으로 다시 돌린 경우: 결과를 NaN으로 덮지 않는다
        vl = validate()
    json.dump({"run": a.run, "scale": scale, "arm": arm, "corpus": corpus, "seed": a.seed,
               "params": n_params, "tokens": total_tokens, "steps": max_steps,
               "val_loss": vl, "wall_s": round(time.time() - t0),
               "device": jax.devices()[0].device_kind, "devices": devs},
              open(res_path, "w"), indent=2)
    print(f"done: {res_path} val {vl:.4f}", flush=True)


if __name__ == "__main__":
    main()
