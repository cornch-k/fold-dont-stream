#!/usr/bin/env python3
"""EXP-38b 게이트 적합: 층 30 릿지 보정의 홀드아웃 잔차.
판정 기준 — 게이팅측 천장(§4.5)의 상대잔차 0.93 대비:
  홀드아웃 잔차 << 0.93  → 가중치측 문이 열림 → 전 층 파이프라인 구축
  홀드아웃 잔차 ~ 0.93+  → §4.5 가 가중치측까지 확장 (그 자체로 결과)"""
import numpy as np, glob, sys

def load(dirpath, name):
    fs = sorted(glob.glob(f"{dirpath}/{name}_*.bin"))
    if not fs: sys.exit(f"덤프 없음: {dirpath}/{name}")
    a = np.concatenate([np.fromfile(f, dtype=np.float32).reshape(-1, 3072) for f in fs])
    return a

X  = load("exp/acts-fold", "attn_residual-30")   # 층30 입력 (두 런에서 동일해야 함)
X2 = load("exp/acts-orig", "attn_residual-30")
Yf = load("exp/acts-fold", "ffn_out-30")
Yo = load("exp/acts-orig", "ffn_out-30")
T = min(len(X), len(X2), len(Yf), len(Yo))
X, X2, Yf, Yo = X[:T], X2[:T], Yf[:T], Yo[:T]

# 전제 검증: 층 0-29 동일 → 입력 동일
drift = np.linalg.norm(X - X2) / (np.linalg.norm(X) + 1e-9)
print(f"토큰 {T}개, 입력 드리프트 {drift:.2e}  (0 이어야 함)")
if drift > 1e-3: sys.exit("!! 입력이 다름 — 맵 전제 위반, 중단")

R = Yo - Yf
print(f"목표 노름비 ||R||/||Yo|| = {np.linalg.norm(R)/np.linalg.norm(Yo):.4f}")

ntr = int(T * 0.75)
Xt, Xh, Rt, Rh = X[:ntr], X[ntr:], R[:ntr], R[ntr:]
G = Xt.T @ Xt
b = Xt.T @ Rt
base = np.linalg.norm(Rh)
print(f"train {ntr} / holdout {T-ntr}")
for lam in [1e0, 1e1, 1e2, 1e3, 1e4]:
    W = np.linalg.solve(G + lam * np.eye(3072, dtype=np.float64), b)
    res = np.linalg.norm(Rh - Xh @ W) / base
    # 저랭크 절단 r=64 도 같이 (배포 형태)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    W64 = (U[:, :64] * S[:64]) @ Vt[:64]
    res64 = np.linalg.norm(Rh - Xh @ W64) / base
    print(f"λ={lam:8.0f}  홀드아웃 잔차 full={res:.4f}  rank64={res64:.4f}")
print("게이팅측 천장(§4.5): 0.93~0.95, 우연 바닥 0.9574")
