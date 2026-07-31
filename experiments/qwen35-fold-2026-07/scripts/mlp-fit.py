#!/usr/bin/env python3
"""EXP-40 비선형 게이트: 3072→512 GELU→3072 MLP 가 접기 잔차 R 을 잡는가.
선형 천장 0.9636 (EXP-38). 이걸 유의하게 뚫으면 → 비선형 어댑터 파이프라인.
못 뚫으면 → R 은 x 의 함수가 아님(사라진 전문가의 정보) → 입력측 수리 전면 폐쇄."""
import numpy as np, glob, sys, time

def load(d, n):
    fs = sorted(glob.glob(f"exp/{d}/{n}_*.bin"))
    return np.concatenate([np.fromfile(f, dtype=np.float32).reshape(-1, 3072) for f in fs])

X = load("acts-fold", "attn_residual-30"); Yf = load("acts-fold", "ffn_out-30")
Yo = load("acts-orig", "ffn_out-30")
T = min(len(X), len(Yf), len(Yo)); X, R = X[:T], (Yo - Yf)[:T]
ntr = int(T * 0.75)
mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
Xn = (X - mu) / sd
Xt, Xh, Rt, Rh = Xn[:ntr], Xn[ntr:], R[:ntr], R[ntr:]
base = np.linalg.norm(Rh)
print(f"T={T} train={ntr} holdout={T-ntr}  선형 천장 0.9636")

rng = np.random.default_rng(0)
D, H = 3072, 512
W1 = rng.normal(0, (2/D)**0.5, (D, H)).astype(np.float64); b1 = np.zeros(H)
W2 = np.zeros((H, D)); b2 = np.zeros(D)   # 출력 0 초기화 → 시작 잔차 = 1.0
mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1); mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2); mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
lr, b1a, b2a, eps, t0 = 1e-3, 0.9, 0.999, 1e-8, time.time()
step = 0
def gelu(z): return 0.5*z*(1+np.tanh(0.79788456*(z+0.044715*z**3)))
def dgelu(z):
    th = np.tanh(0.79788456*(z+0.044715*z**3))
    return 0.5*(1+th) + 0.5*z*(1-th**2)*0.79788456*(1+3*0.044715*z**2)
best = 9e9
for ep in range(60):
    idx = rng.permutation(ntr)
    for s in range(0, ntr, 512):
        bidx = idx[s:s+512]; x, r = Xt[bidx], Rt[bidx]
        z = x @ W1 + b1; a = gelu(z); p = a @ W2 + b2
        e = (p - r) / len(bidx)
        gW2 = a.T @ e; gb2 = e.sum(0)
        da = e @ W2.T; dz = da * dgelu(z)
        gW1 = x.T @ dz; gb1 = dz.sum(0)
        step += 1
        for (P, g, m, v) in ((W1,gW1,mW1,vW1),(b1,gb1,mb1,vb1),(W2,gW2,mW2,vW2),(b2,gb2,mb2,vb2)):
            m *= b1a; m += (1-b1a)*g
            v *= b2a; v += (1-b2a)*g*g
            P -= lr * (m/(1-b1a**step)) / (np.sqrt(v/(1-b2a**step)) + eps)
    if ep % 5 == 4 or ep == 0:
        ph = gelu(Xh @ W1 + b1) @ W2 + b2
        res = np.linalg.norm(Rh - ph) / base
        best = min(best, res)
        print(f"ep{ep+1:3d}  홀드아웃 잔차 {res:.4f}   ({time.time()-t0:.0f}s)")
print(f"최선 {best:.4f}  vs 선형 0.9636 / §4.5 천장 0.93~0.957")
