#!/usr/bin/env python3
"""EXP-41b 손상 랭킹 + 손상 기반 후보 맵 생성.
damage_il = ||ffn_out_orig(il) − ffn_out_fold(il)|| / ||ffn_out_orig(il)||
후보 맵: 손상 상위 K개 층을 대표(=자기 자신 상주)로 승격, 나머지는 가장 가까운 앞쪽 대표에 접기.
예산 = 상주 슬롯 11 (고정 44-47 4개 + 자유 7개) 유지."""
import numpy as np, glob, sys

def load(d, n):
    fs = sorted(glob.glob(f"exp/{d}/{n}_*.bin"))
    if not fs: return None
    return np.concatenate([np.fromfile(f, dtype=np.float32).reshape(-1, 3072) for f in fs])

# 현행 맵 재구성
def curmap():
    m = []
    for il in range(48):
        if il >= 44: m.append(il)
        else:
            g = il * 7 // 44
            m.append((g * 44 + 6) // 7)
    return m

M = curmap()
folded = [il for il in range(44) if M[il] != il]
rows = []
for il in folded:
    yo = load(f"cen-{il}", f"ffn_out-{il}")
    yf = load("cen-base", f"ffn_out-{il}")
    if yo is None or yf is None:
        print(f"il={il}: 덤프 누락"); continue
    T = min(len(yo), len(yf)); yo, yf = yo[:T], yf[:T]
    d = np.linalg.norm(yo - yf) / (np.linalg.norm(yo) + 1e-9)
    rows.append((il, d))
rows.sort(key=lambda r: -r[1])
print("손상 랭킹 (상위 15):")
for il, d in rows[:15]: print(f"  il={il:2d}  {d:.4f}")
print("하위 5:")
for il, d in rows[-5:]: print(f"  il={il:2d}  {d:.4f}")

# 후보 맵: 자유 슬롯 7개를 손상 상위 7개로
top7 = sorted([il for il, _ in rows[:7]])
reps = sorted(set([0] + top7))[:7]  # 층0은 dense-lead 관례상 후보에 포함해 비교
# 실제로는 상위 7 그대로: 두 변형 생성
def build(reps):
    reps = sorted(set(reps))
    m = []
    for il in range(48):
        if il >= 44: m.append(il)
        elif il in reps: m.append(il)
        else:
            prev = [r for r in reps if r <= il]
            m.append(max(prev) if prev else min(reps))
    return m
for name, rset in [("damage7", top7), ("damage6+0", [0] + sorted([il for il, _ in rows[:6]]))]:
    mm = build(rset)
    uniq = sorted(set(mm))
    print(f"\n{name}: 대표 {rset}")
    print(f"  MAP={','.join(map(str, mm))}")
    print(f"  유일층 {len(uniq)}개: {uniq}")
