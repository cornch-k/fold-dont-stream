#!/usr/bin/env python3
import argparse, math, os, time, csv
import numpy as np, torch
from model import GPT, CONFIGS

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, EFF_BATCH_TOK, WARMUP, LR, LR_MIN, WD, CLIP = 1337, 32768, 300, 6e-4, 6e-5, 0.1, 1.0
EVAL_EVERY, EVAL_BATCHES = 500, 100
AUX = 0.01  # MoE load-balance 계수

def get_batch(arr, B, T, gen):
    ix = torch.randint(0, len(arr) - T - 1, (B,), generator=gen)
    x = torch.stack([torch.from_numpy(arr[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(arr[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
    return x, y

def val_loss(model, arr, B, T, dev, amp):
    model.eval()
    if getattr(model.cfg, "use_moe", False):
        model.reset_expert_stats(True)
    losses = []
    with torch.no_grad():
        for k in range(EVAL_BATCHES):
            s = k * B * T
            if s + B * T + 1 > len(arr):
                break
            x = torch.stack([torch.from_numpy(arr[s + j * T: s + (j + 1) * T].astype(np.int64)) for j in range(B)])
            y = torch.stack([torch.from_numpy(arr[s + j * T + 1: s + (j + 1) * T + 1].astype(np.int64)) for j in range(B)])
            with amp:
                _, l = model(x.to(dev), y.to(dev))
            losses.append(l.item())
    model.train()
    return sum(losses) / len(losses)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(CONFIGS))
    ap.add_argument("--max-tokens", type=int, default=200_000_000)
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    cfg = CONFIGS[a.run]
    model = GPT(cfg).to(dev)
    print(f"run={a.run} unique_params={model.num_unique_params():,} "
          f"resident_fp16={model.resident_bytes()/1e6:.1f}MB dev={dev} dtype={a.dtype}", flush=True)
    T, B = cfg.ctx, a.micro_batch
    accum = max(1, EFF_BATCH_TOK // (B * T))
    train_arr = np.memmap(os.path.join(HERE, "data/train.bin"), dtype=np.uint16, mode="r")
    val_arr = np.memmap(os.path.join(HERE, "data/val.bin"), dtype=np.uint16, mode="r")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WD)
    amp = (torch.autocast(device_type=dev, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype])
           if a.dtype != "fp32" else torch.autocast(device_type=dev, enabled=False))
    scaler = torch.amp.GradScaler("cuda") if (dev == "cuda" and a.dtype == "fp16") else None
    gen = torch.Generator().manual_seed(a.seed)
    max_steps = a.max_tokens // EFF_BATCH_TOK
    step = 0
    out_dir = os.environ.get("RDEPTH_OUT", HERE)
    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    sfx = f"-s{a.seed}" if a.seed != 1337 else ""
    ckpt_path = os.path.join(out_dir, f"ckpt/{a.run}{sfx}.pt")
    log_path = os.path.join(out_dir, f"logs/{a.run}{sfx}.csv")
    if a.resume and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # gen 상태는 CPU ByteTensor여야 함
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        gen.set_state(st["gen"])
        print(f"resumed at step {step}", flush=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "tokens", "train_loss", "val_loss", "tok_s", "elapsed_s", "uniq_exp"])
    t0, tok_count = time.time(), 0
    while step < max_steps:
        lr = (LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * min(1.0, step / max_steps)))
              if step >= WARMUP else LR * (step + 1) / WARMUP)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            x, y = get_batch(train_arr, B, T, gen)
            with amp:
                _, loss = model(x.to(dev), y.to(dev))
            total = loss if getattr(model, "last_aux", None) is None else loss + AUX * model.last_aux
            if scaler is not None:
                scaler.scale(total / accum).backward()
            else:
                (total / accum).backward()
            loss_acc += loss.item() / accum
        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        step += 1
        tok_count += EFF_BATCH_TOK
        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step}/{max_steps} loss {loss_acc:.4f} {tok_count/el:,.0f} tok/s", flush=True)
        if step % EVAL_EVERY == 0 or step == max_steps:
            vl = val_loss(model, val_arr, B, T, dev, amp)
            ue = f"{model.expert_stats():.3f}" if cfg.use_moe else ""
            model.reset_expert_stats(False)
            el = time.time() - t0
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, step * EFF_BATCH_TOK, f"{loss_acc:.4f}", f"{vl:.4f}",
                                        f"{tok_count/el:.0f}", f"{el:.0f}", ue])
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "gen": gen.get_state()}, ckpt_path)
            print(f"  [eval] step {step} val {vl:.4f}" + (f" uniq_exp {ue}" if ue else ""), flush=True)

if __name__ == "__main__":
    main()
