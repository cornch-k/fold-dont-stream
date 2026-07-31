#!/usr/bin/env python3
"""TinyStories → ByteLevel BPE 4096 → uint16 memmap bins. EOT token id 0."""
import os, sys, numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EOT = "<|endoftext|>"

def download():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    paths = {}
    for split, fn in [("train", "TinyStories-train.txt"), ("val", "TinyStories-valid.txt")]:
        paths[split] = hf_hub_download("roneneldan/TinyStories", fn, repo_type="dataset")
    return paths

def train_tokenizer(train_path, sample_mb=200):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    def sample_iter():
        read = 0
        with open(train_path, encoding="utf-8") as f:
            for line in f:
                read += len(line)
                if read > sample_mb * 1_000_000: break
                yield line
    tok.train_from_iterator(sample_iter(), trainer)
    tok.save(os.path.join(DATA, "tok4096.json"))
    return tok

def encode_split(tok, txt_path, out_path):
    eot_id = tok.token_to_id(EOT)
    assert eot_id == 0, f"EOT id must be 0, got {eot_id}"
    buf, chunks = [], []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == EOT:
                if buf:
                    ids = tok.encode("".join(buf)).ids
                    chunks.append(np.array(ids + [eot_id], dtype=np.uint16))
                    buf = []
            else:
                buf.append(line)
    if buf:
        chunks.append(np.array(tok.encode("".join(buf)).ids + [eot_id], dtype=np.uint16))
    arr = np.concatenate(chunks)
    arr.tofile(out_path)
    print(f"{out_path}: {len(arr):,} tokens")

if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    p = download()
    print("downloaded:", p)
    tok_path = os.path.join(DATA, "tok4096.json")
    tok = Tokenizer.from_file(tok_path) if os.path.exists(tok_path) else train_tokenizer(p["train"])
    print("tokenizer ready")
    for split in ["val", "train"]:  # val 먼저 (빠른 실패)
        out = os.path.join(DATA, f"{split}.bin")
        if not os.path.exists(out):
            encode_split(tok, p[split], out)
