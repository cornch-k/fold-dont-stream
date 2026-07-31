#!/usr/bin/env python3
"""results/*.json → 회복률 표 + 시드 집계 + 스케일 추세선.

recovery = (val_small − val_loop) / (val_small − val_large)
         = 2× 메모리로 얻는 품질 격차 중 looping이 되찾는 비율.

비율이라 노이즈가 증폭된다. loop arm이 0.005 움직이면 recovery는 0.055 움직인다.
그래서 시드별 recovery를 따로 낸 뒤 평균 ± 표준편차로 묶는다.

  python scaling.py --out-dir . --plot results/recovery.png
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

GATE = (0.41, 0.04)      # PyTorch 15M TinyStories 참조: 1.5950/1.5577/1.5038 → 0.409
ORDER = ["15m", "50m", "150m"]
ARMS = ("small", "loop", "large")


def recovery(arms):
    s, l, g = (arms[a]["val_loss"] for a in ARMS)
    gap = s - g
    return (s - l) / gap, gap


def mean_sd(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--plot", default="")
    a = ap.parse_args()

    runs = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(a.out_dir, "results", "*.json"))):
        r = json.load(open(p))
        if r.get("val_loss") is None or math.isnan(r["val_loss"]):
            print(f"[경고] {os.path.basename(p)} val_loss 없음 — 건너뜀")
            continue
        runs[(r["corpus"], r["scale"], r["seed"])][r["arm"]] = r

    def key(k):
        return (k[0], ORDER.index(k[1]) if k[1] in ORDER else 9, k[2])

    per_seed = defaultdict(list)      # (corpus, scale) → [(seed, recovery, gap, params)]
    print(f"{'corpus':<12}{'scale':<7}{'seed':>6}{'params':>12}  "
          f"{'small':>7}{'loop':>8}{'large':>8}{'gap':>8}{'recovery':>10}")
    for k in sorted(runs, key=key):
        arms = runs[k]
        missing = set(ARMS) - set(arms)
        if missing:
            print(f"{k[0]:<12}{k[1]:<7}{k[2]:>6}  미완: {sorted(missing)}")
            continue
        rec, gap = recovery(arms)
        n = arms["small"]["params"]
        print(f"{k[0]:<12}{k[1]:<7}{k[2]:>6}{n:>12,}  {arms['small']['val_loss']:>7.4f}"
              f"{arms['loop']['val_loss']:>8.4f}{arms['large']['val_loss']:>8.4f}"
              f"{gap:>8.4f}{rec:>10.4f}")
        per_seed[(k[0], k[1])].append((k[2], rec, gap, n))

    print(f"\n{'corpus':<12}{'scale':<7}{'params':>12}{'n':>4}{'recovery':>10}{'sd':>9}")
    agg = {}
    for ck in sorted(per_seed, key=lambda c: (c[0], ORDER.index(c[1]) if c[1] in ORDER else 9)):
        rows = per_seed[ck]
        m, sd = mean_sd([r[1] for r in rows])
        n = rows[0][3]
        agg[ck] = (n, len(rows), m, sd)
        print(f"{ck[0]:<12}{ck[1]:<7}{n:>12,}{len(rows):>4}{m:>10.4f}"
              + (f"{sd:>9.4f}" if len(rows) > 1 else f"{'—':>9}"))

    for (corpus, scale), (n, cnt, m, sd) in agg.items():
        if scale == "15m" and corpus.startswith("tinystories"):
            ok = abs(m - GATE[0]) <= GATE[1]
            print(f"\n[게이트] 15m/{corpus} recovery={m:.4f} (n={cnt}) "
                  f"목표 {GATE[0]}±{GATE[1]} → {'통과' if ok else '실패 — Phase 2 진행 금지'}")
    for ck, rows in per_seed.items():
        g = min(r[2] for r in rows)
        if g < 0.02:
            print(f"[경고] {ck[0]}/{ck[1]} small−large 격차 {g:.4f} nats. "
                  "분모가 시드 노이즈에 가까워 회복률을 신뢰할 수 없습니다.")

    ladder = sorted(((v[0], v[2], v[3], v[1]) for k, v in agg.items()
                     if not k[0].startswith("tinystories")))
    if a.plot and len(ladder) >= 2:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.array([r[0] for r in ladder], float)
        y = np.array([r[1] for r in ladder], float)
        e = np.array([0.0 if math.isnan(r[2]) else r[2] for r in ladder], float)
        k, b = np.polyfit(np.log10(x), y, 1)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.errorbar(x, y, yerr=e, fmt="o-", capsize=3, label="dense (FineWeb-Edu)")
        for xi, _, _, ni in ladder:
            ax.annotate(f"n={ni}", (xi, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7,
                        color="gray")
        xs = np.logspace(np.log10(x.min()) - 0.1, np.log10(x.max()) + 0.3, 50)
        ax.semilogx(xs, k * np.log10(xs) + b, "--", lw=1, label=f"fit: {k:+.3f} / decade")
        ax.axhline(GATE[0], color="gray", lw=0.8, ls=":", label="15M TinyStories (PyTorch)")
        ax.set_xscale("log")
        ax.set_xlabel("unique parameters"), ax.set_ylabel("recovery")
        ax.legend(fontsize=8), ax.grid(alpha=0.3)
        fig.tight_layout(), fig.savefig(a.plot, dpi=160)
        sd = [r[2] for r in ladder if not math.isnan(r[2])]
        print(f"\n{a.plot} 저장. 기울기 {k:+.4f} / 10배"
              + (f" (점당 시드 산포 {min(sd):.3f}~{max(sd):.3f})" if sd else ""))


if __name__ == "__main__":
    main()
