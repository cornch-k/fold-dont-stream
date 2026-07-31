#!/usr/bin/env python3
"""MLX 접힌 모델의 wikitext ppl. llama.cpp perplexity.cpp 프로토콜 그대로:
텍스트를 n_ctx 청크로 자르고 각 청크의 후반부(first = n_ctx/2 부터)만 채점한다.
BOS 없음(Qwen). 게이트: llama.cpp 접힌 A 구성 14.6569 와 대조(MLX 재양자화 오차만큼 상향 허용).
"""
import argparse, math, time, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
import fold_model

AP = argparse.ArgumentParser()
AP.add_argument("--path", default="heal/mlx-fold")
AP.add_argument("--text", default="nanfix/scripts/wikitext-2-raw/wiki.test.raw")
AP.add_argument("--ctx", type=int, default=512)
AP.add_argument("--chunks", type=int, default=16)
AP.add_argument("--delta", type=float, default=None)
AP.add_argument("--adapter", default=None, help="힐링 어댑터 safetensors")
A = AP.parse_args()

t0 = time.time()
model, args, cfg = fold_model.load(A.path, A.delta)
print(f"로드 {time.time()-t0:.0f}s", flush=True)
if A.adapter:
    import heal_train
    heal_train.attach(model, A.adapter)
    print(f"어댑터 적용: {A.adapter}", flush=True)
model.eval()

tok = AutoTokenizer.from_pretrained("heal/tok")
ids = tok(open(A.text).read()).input_ids
print(f"토큰 {len(ids)}개, 청크 {A.chunks}×{A.ctx}", flush=True)

first = A.ctx // 2
nlls, n = [], 0
for c in range(A.chunks):
    chunk = ids[c * A.ctx:(c + 1) * A.ctx]
    if len(chunk) < A.ctx:
        break
    x = mx.array([chunk])
    logits = model(x).astype(mx.float32)
    # 위치 i 의 logits 가 토큰 i+1 을 예측. first..ctx-1 번째 토큰만 채점.
    lp = (logits[0, first - 1:-1] - mx.logsumexp(logits[0, first - 1:-1], axis=-1, keepdims=True))
    tgt = mx.array(chunk[first:])
    nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1).sum()
    mx.eval(nll)
    nlls.append(float(nll)); n += len(chunk) - first
    del x, logits, lp
    mx.clear_cache()
    print(f"  [{c+1}/{A.chunks}] ppl {math.exp(sum(nlls)/n):.4f}  ({time.time()-t0:.0f}s)", flush=True)

ppl = math.exp(sum(nlls) / n)
# 청크별 표준오차 (llama.cpp 와 동일한 방식은 아니지만 산포 참고용)
per = [x / (A.ctx - first) for x in nlls]
se = float(np.std(per, ddof=1) / math.sqrt(len(per))) * ppl if len(per) > 1 else 0.0
print(f"최종 ppl {ppl:.4f} ± {se:.4f}   ({n}토큰, {time.time()-t0:.0f}s)")
