#!/usr/bin/env python3
"""ppl ↔ 벤치 전이율. 로그에서 직접 읽은 값만 쓴다 (추정 금지).

유효 쌍 = 같은 구성에서 ppl 과 벤치가 모두 측정된 경우.
표본이 적으므로 상관계수는 보고하지 않고(n=2 면 항등식으로 ±1), 대신
  · 구성 간 델타를 벤치 측정오차와 비교 (오차 안/밖)
  · 최소제곱 기울기와 그 표준오차 (n>=3 일 때만)
를 낸다.
"""
import re, sys, os, math
import numpy as np

# (구성, ppl 로그, ppl 오차, 벤치 로그들) — 전부 실측 파일 경로
CONFIGS = [
    # 구성 C: v2a 11층 + γg32 + k6
    dict(tag="C(11층)", ppl="exp/e32-ppl.log",
         mmlu="exp/bf-folded-mmlu.log", hswag="exp/mk-folded-hswag.log",
         wino="exp/mk-folded-wino.log", arc="exp/mk-folded-arc.log"),
    # 구성 A: v2a-hot 6rep 공식맵 + δ1.0 + γg46 + k6
    dict(tag="A(6rep+hot16)", ppl="exp/e50-ppl.log",
         mmlu="exp/e53-mmlu2000.log", hswag="exp/e53-hswag.log",
         wino="exp/e53-wino.log", arc="exp/e53-arc.log"),
    # 구성 Q3: 대표 22,26,30,32,38,41 (앞쪽 예산 재배치)
    dict(tag="Q3(22,26,30,32,38,41)", ppl="exp/e58-Q3-ppl.log",
         mmlu="exp/e60-Q3-mmlu.log", hswag="exp/e60-Q3-hswag.log",
         wino="exp/e60-Q3-wino.log", arc="exp/e60-Q3-arc.log"),
]


def scan(path, pat, group=1, err=None):
    if not path or not os.path.exists(path):
        return None
    val = errv = None
    with open(path, 'rb') as f:
        for line in f:
            m = re.search(pat, line)
            if m:
                val = float(m.group(group))
                errv = float(m.group(err)) if err else None
    return None if val is None else (val, errv)


def mc_partial(path):
    """Final result 가 없는(중단된) MC 로그: 마지막 '태스크수\t정답률' 줄에서 읽고 이항오차 부여."""
    if not path or not os.path.exists(path):
        return None
    last = None
    with open(path, 'rb') as f:
        for line in f:
            m = re.match(rb'^(\d+)\t([\d.]+)\s*$', line)
            if m:
                last = (int(m.group(1)), float(m.group(2)))
    if not last:
        return None
    n, acc = last
    se = 100.0 * math.sqrt((acc / 100) * (1 - acc / 100) / n)
    return (acc, se, n)


def read(cfg):
    out = {"tag": cfg["tag"]}
    out["ppl"] = scan(cfg["ppl"], rb'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)', 1, 2)
    out["mmlu"] = scan(cfg["mmlu"], rb'Final result: ([\d.]+) \+/- ([\d.]+)', 1, 2)
    out["mmlu_n"] = 2000
    if out["mmlu"] is None:
        pv = mc_partial(cfg["mmlu"])
        if pv:
            out["mmlu"], out["mmlu_n"] = (pv[0], pv[1]), pv[2]
    hs = scan(cfg["hswag"], rb'^\s*(?:399|400)\s+([\d.]+)%')
    out["hswag"] = (hs[0], None) if hs else None
    out["wino"] = scan(cfg["wino"], rb'Final Winogrande score\(\d+ tasks\): ([\d.]+) \+/- ([\d.]+)', 1, 2)
    out["arc"] = scan(cfg["arc"], rb'Final result: ([\d.]+) \+/- ([\d.]+)', 1, 2)
    if out["arc"] is None:
        pv = mc_partial(cfg["arc"])
        if pv: out["arc"] = (pv[0], pv[1])
    return out


rows = [read(c) for c in CONFIGS]
BENCH = [("mmlu", "MMLU-2000"), ("hswag", "HellaSwag-400"), ("wino", "Winogrande"), ("arc", "ARC-C")]

print("── 유효 쌍 ──")
valid = []
for r in rows:
    have = [k for k, _ in BENCH if r[k]]
    ok = r["ppl"] is not None and len(have) == len(BENCH)
    print(f"{r['tag']:26s} ppl={r['ppl'][0] if r['ppl'] else '없음'}  벤치 {len(have)}/4 {'✓' if ok else '(불완전)'}")
    if ok:
        valid.append(r)
n = len(valid)
print(f"\n유효 쌍 n={n}")
if n < 2:
    sys.exit("n<2 — 분석 불가")

print("\n── 값 ──")
hdr = f"{'구성':26s} {'ppl':>8s}" + "".join(f"{name:>18s}" for _, name in BENCH)
print(hdr)
for r in valid:
    line = f"{r['tag']:26s} {r['ppl'][0]:8.3f}"
    for k, _ in BENCH:
        v, e = r[k]
        line += f"{v:12.2f}±{e:4.2f}" if e else f"{v:17.2f}"
    print(line)

print("\n── 전이율 (ppl 1 감소당 벤치 점수 변화) ──")
ppl = np.array([r["ppl"][0] for r in valid])
for k, name in BENCH:
    y = np.array([r[k][0] for r in valid])
    errs = [r[k][1] for r in valid]
    if n >= 3:
        A = np.vstack([ppl, np.ones(n)]).T
        beta, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope = -beta[0]                     # ppl 감소당이므로 부호 반전
        yhat = A @ beta
        dof = n - 2
        s2 = float(((y - yhat) ** 2).sum() / dof) if dof > 0 else 0.0
        cov = s2 * np.linalg.inv(A.T @ A)
        se = math.sqrt(abs(cov[0, 0]))
        print(f"{name:16s} 기울기 {slope:+7.3f} 점/ppl  (SE {se:.3f}, n={n}, dof={dof})", end="")
    else:
        slope = -(y[-1] - y[0]) / (ppl[-1] - ppl[0])
        print(f"{name:16s} 기울기 {slope:+7.3f} 점/ppl  (2점 secant, SE 계산불가)", end="")
    # 양 끝 구성 간 델타가 오차를 벗어나는가
    d = y[-1] - y[0]
    e0, e1 = errs[0], errs[-1]
    if e0 and e1:
        comb = math.sqrt(e0 ** 2 + e1 ** 2)
        print(f"   Δ={d:+.2f} vs 합성오차 {comb:.2f} → {'오차 밖(유의)' if abs(d) > comb else '오차 안'}")
    else:
        print(f"   Δ={d:+.2f} (오차 미기록)")

print("\n── MMLU 40 외삽 ──")
y = np.array([r["mmlu"][0] for r in valid])
if n >= 3:
    A = np.vstack([ppl, np.ones(n)]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = -beta[0]
else:
    slope = -(y[-1] - y[0]) / (ppl[-1] - ppl[0])
need = y[-1] + 0.0
gap = 40.0 - need
if abs(slope) < 1e-9:
    print("기울기 ~0 — ppl 축으로 외삽 불가")
else:
    target = ppl[-1] - gap / slope
    print(f"현재 MMLU {need:.2f} @ ppl {ppl[-1]:.3f}, 기울기 {slope:+.3f} → MMLU 40 필요 ppl = {target:.2f}")
    print(f"  (무접기 실측 바닥 5.095 대비 {'도달 가능' if target >= 5.095 else '바닥보다 낮음 → 이 축만으로는 불가'})")
