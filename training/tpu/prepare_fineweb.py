#!/usr/bin/env python3
"""FineWeb-Edu → 4k ByteLevel BPE → uint16 memmap bins. EOT token id 0.

training/rdepth/prepare_data.py와 동일한 토크나이저 절차. vocab을 4096으로
유지하는 이유: GPT-2 어휘(50257)를 쓰면 d=512에서 임베딩만 25.7M 파라미터라
15M 사다리 칸이 성립하지 않는다. 세 스케일이 같은 토크나이저를 쓰므로 13:1
토큰 비율도 전부 이 토큰 단위로 센다.

  python prepare_fineweb.py --out data/fineweb --target-tokens 1900000000

150M arm이 1.863B 토큰을 요구하므로 기본값 1.9B. 결과 train.bin ≈ 3.8 GB.
"""
import argparse
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

EOT = "<|endoftext|>"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def stream(sample):
    return load_dataset("HuggingFaceFW/fineweb-edu", name=sample, split="train", streaming=True)


def train_tokenizer(sample, out, sample_mb=200):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def it():
        read = 0
        for doc in stream(sample):
            read += len(doc["text"])
            if read > sample_mb * 1_000_000:
                return
            yield doc["text"]

    tok.train_from_iterator(it(), trainer)
    tok.save(out)
    return tok


def encode_to(tok, docs, path, target, batch=1000):
    """docs 이터레이터를 target 토큰까지 인코딩해 path에 append. 실제 토큰 수 반환."""
    eot = tok.token_to_id(EOT)
    assert eot == 0, f"EOT id must be 0, got {eot}"
    n, buf = 0, []
    with open(path, "wb") as f:
        for doc in docs:
            buf.append(doc["text"])
            if len(buf) < batch:
                continue
            for enc in tok.encode_batch(buf):
                arr = np.array(enc.ids + [eot], dtype=np.uint16)
                f.write(arr.tobytes())
                n += len(arr)
            buf = []
            if n >= target:
                break
            if n % 100_000_000 < batch * 512:
                print(f"  {path}: {n:,} tokens", flush=True)
    print(f"{path}: {n:,} tokens", flush=True)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fineweb")
    ap.add_argument("--sample", default="sample-10BT")
    ap.add_argument("--target-tokens", type=int, default=1_900_000_000)
    ap.add_argument("--val-tokens", type=int, default=5_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tok_path = os.path.join(a.out, "tok4096.json")
    tok = (Tokenizer.from_file(tok_path) if os.path.exists(tok_path)
           else train_tokenizer(a.sample, tok_path))
    print("tokenizer ready", flush=True)

    # val은 스트림 앞부분, train은 그 뒤 — 같은 이터레이터라 문서가 겹치지 않는다
    docs = iter(stream(a.sample))
    if not os.path.exists(os.path.join(a.out, "val.bin")):
        encode_to(tok, docs, os.path.join(a.out, "val.bin"), a.val_tokens)
    else:                       # val을 건너뛰어도 train이 같은 문서를 안 보게 소진
        for _ in range(a.val_tokens // 400):
            next(docs, None)
    if not os.path.exists(os.path.join(a.out, "train.bin")):
        encode_to(tok, docs, os.path.join(a.out, "train.bin"), a.target_tokens)
