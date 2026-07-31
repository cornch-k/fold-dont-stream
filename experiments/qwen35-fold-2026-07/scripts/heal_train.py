#!/usr/bin/env python3
"""접힌 122B 엔드투엔드 LM-loss 힐링.

왜 이게 앞선 실패들과 다른가: 앞서 닫힌 수리(γ 스칼라·게이팅·선형 d²·비선형 MLP)는 모두
**층별 재구성** 목표를 최소화했다. 그런데 이 프로젝트가 세 번 실측한 사실이 "층별 지표는
엔드투엔드와 반상관"이다. 즉 층별 목표로 닫힌 문은 엔드투엔드 목표에 대해 닫혀 있지 않다.
여기서는 최종 LM loss 의 기울기만 쓴다.

학습 대상(전부 GGUF 로 되돌릴 수 있는 텐서만 — 추론 비용 증가 0):
  · 논리층별 FFN pre-norm  (LLAMA_FOLD_NORM_LOGICAL 노브로 배포, 3072 float/층)
  · 논리층별 attn pre-norm
  · 논리층별 공유전문가 LoRA (병합 후 q6_K 재양자화)
  · 논리층별 공유전문가 게이트 / 핫 라우터
  · 물리층 라우터
동결: 전문가 본체(3bit), 핫 전문가 본체, 어텐션/SSM 전체.
"""
import argparse, json, math, os, sys, time, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.lora import LoRALinear
from transformers import AutoTokenizer
import fold_model

SHEXP = ("gate_proj", "up_proj", "down_proj")


def grad_checkpoint(model):
    """층마다 nn.utils.checkpoint 래퍼를 달아 활성값 대신 재연산으로 메모리를 산다.
    래퍼는 그 층의 trainable_parameters 만 미분 트리로 잡는다 — 그래서 학습 대상은
    전부 **논리층 자기 것**이어야 한다. 물리층 공유 라우터를 학습 대상에 넣으면
    접힌 층 경로의 기울기가 조용히 누락된다(그래서 라우터는 동결)."""
    for layer in model.layers:
        layer._ckpt = nn.utils.checkpoint(layer)


def wrap_lora(model, r, scale):
    n = 0
    for layer in model.layers:
        se = layer.mlp.shared_expert
        for name in SHEXP:
            setattr(se, name, LoRALinear.from_base(getattr(se, name), r=r, scale=scale))
            n += 1
    return n


def set_trainable(model):
    """전부 동결 후 대상만 해제. LoRA 는 lora_a/lora_b 만 학습 가능해진다."""
    model.freeze()
    full = 0
    for layer in model.layers:
        layer.input_layernorm.unfreeze()
        layer.post_attention_layernorm.unfreeze()
        layer.mlp.shared_expert_gate.unfreeze()
        if "hot_gate" in layer.mlp:
            layer.mlp.hot_gate.unfreeze()
        # 물리층 라우터(layer.mlp.gate)는 동결: 층별 체크포인트가 공유 모듈의 기울기를
        # 잡지 못하고, 여러 논리층이 공유하므로 변경 위험도 크다.
        for name in SHEXP:
            lin = getattr(layer.mlp.shared_expert, name)
            lin.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    return full


def adapter_dict(model):
    return {k: v for k, v in tree_flatten(model.trainable_parameters())}


def attach(model, path):
    """저장된 어댑터를 모델에 적용 (ppl.py 평가용)."""
    w = mx.load(path)
    if any(".lora_a" in k for k in w):
        cfg = json.load(open(os.path.join(os.path.dirname(path), "adapter_config.json")))
        wrap_lora(model, cfg["r"], cfg["scale"])
    model.update(tree_unflatten(list(w.items())))
    mx.eval(model.parameters())


def batches(ids, ctx, bs, rng):
    """무작위 시작점 청크. 청크 경계 편향을 줄인다."""
    while True:
        starts = rng.integers(0, len(ids) - ctx - 1, size=bs)
        yield mx.array(np.stack([ids[s:s + ctx + 1] for s in starts]))


def loss_fn(model, batch):
    logits = model(batch[:, :-1]).astype(mx.float32)
    return nn.losses.cross_entropy(logits, batch[:, 1:], reduction="mean")


def main():
    AP = argparse.ArgumentParser()
    AP.add_argument("--path", default="heal/mlx-fold")
    AP.add_argument("--out", default="heal/adapter")
    AP.add_argument("--train-text", default="nanfix/scripts/wikitext-2-raw/wiki.train.raw")
    AP.add_argument("--val-text", default="nanfix/scripts/wikitext-2-raw/wiki.valid.raw")
    AP.add_argument("--ctx", type=int, default=512)
    AP.add_argument("--batch", type=int, default=1)
    AP.add_argument("--accum", type=int, default=4)
    AP.add_argument("--steps", type=int, default=600)
    AP.add_argument("--lr", type=float, default=1e-4)
    AP.add_argument("--warmup", type=int, default=20)
    AP.add_argument("--rank", type=int, default=32)
    AP.add_argument("--lora-scale", type=float, default=8.0)
    AP.add_argument("--eval-every", type=int, default=50)
    AP.add_argument("--eval-chunks", type=int, default=8)
    AP.add_argument("--seed", type=int, default=1337)
    AP.add_argument("--resume", default=None)
    A = AP.parse_args()

    os.makedirs(A.out, exist_ok=True)
    t0 = time.time()
    model, args, cfg = fold_model.load(A.path)
    print(f"모델 로드 {time.time()-t0:.0f}s", flush=True)

    nl = wrap_lora(model, A.rank, A.lora_scale)
    set_trainable(model)
    # 층 단위 재계산(gradient checkpointing): 활성값 대신 재연산으로 메모리를 산다.
    # 클래스의 __call__ 을 갈아끼우므로 한 번 호출로 48개 층 전부에 걸린다.
    grad_checkpoint(model)
    if A.resume:
        model.update(tree_unflatten(list(mx.load(A.resume).items())))
    mx.eval(model.parameters())

    tp = adapter_dict(model)
    ntr = sum(v.size for v in tp.values())
    print(f"LoRA {nl}개, 학습 파라미터 {ntr/1e6:.2f}M / 전체 동결 기반", flush=True)
    json.dump({"r": A.rank, "scale": A.lora_scale, "targets": sorted(tp)[:4],
               "n_trainable": int(ntr)}, open(f"{A.out}/adapter_config.json", "w"), indent=1)

    tok = AutoTokenizer.from_pretrained("heal/tok")
    tr = np.array(tok(open(A.train_text).read()).input_ids, dtype=np.int32)
    va = np.array(tok(open(A.val_text).read()).input_ids, dtype=np.int32)
    print(f"학습 {len(tr)}토큰 / 검증 {len(va)}토큰", flush=True)

    sched = optim.join_schedules(
        [optim.linear_schedule(0.0, A.lr, A.warmup),
         optim.cosine_decay(A.lr, max(A.steps - A.warmup, 1), A.lr * 0.1)], [A.warmup])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)
    lvg = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(A.seed)
    gen = batches(tr, A.ctx, A.batch, rng)

    def val_ppl():
        first, tot, n = A.ctx // 2, 0.0, 0
        for c in range(A.eval_chunks):
            ch = va[c * A.ctx:(c + 1) * A.ctx + 1]
            if len(ch) < A.ctx + 1:
                break
            lg = model(mx.array(ch[None, :-1])).astype(mx.float32)
            lp = (lg[0, first - 1:] - mx.logsumexp(lg[0, first - 1:], axis=-1, keepdims=True))
            tgt = mx.array(ch[first:])          # ch 는 ctx+1 개, lp 와 길이 일치
            nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1).sum()
            mx.eval(nll)
            tot += float(nll); n += tgt.size
            mx.clear_cache()
        return math.exp(tot / n)

    base = val_ppl()
    print(f"[step 0] 검증 ppl {base:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    best, hist = base, []

    for step in range(1, A.steps + 1):
        acc, tl = None, 0.0
        for _ in range(A.accum):
            loss, grads = lvg(model, next(gen))
            acc = grads if acc is None else \
                tree_unflatten([(k, a + b) for (k, a), (_, b) in
                                zip(tree_flatten(acc), tree_flatten(grads))])
            tl += float(loss)
            del grads
        acc = tree_unflatten([(k, v / A.accum) for k, v in tree_flatten(acc)])
        opt.update(model, acc)
        mx.eval(model.parameters(), opt.state)
        del acc
        mx.clear_cache()
        tl /= A.accum
        hist.append(tl)
        if step % 10 == 0:
            print(f"[step {step}] loss {np.mean(hist[-10:]):.4f}  lr {float(sched(step)):.2e}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        if step % A.eval_every == 0:
            p = val_ppl()
            print(f"[step {step}] 검증 ppl {p:.4f}  (기준 {base:.4f})", flush=True)
            if p < best:
                best = p
                mx.save_safetensors(f"{A.out}/adapter.safetensors", adapter_dict(model))
                print(f"  → 저장 (최고 {best:.4f})", flush=True)
    print(f"완료: 검증 ppl {base:.4f} → {best:.4f}  ({time.time()-t0:.0f}s)")
    print("힐링 학습 종료")


if __name__ == "__main__":
    main()
