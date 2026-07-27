#!/usr/bin/env python3
"""접힌 모델이 실제로 보는 FFN 입력이, 층별 회귀가 가정한 무접기 입력과 얼마나 다른가.

정렬 검증 내장: 접히지 않은 층(자기 자신에 사상)은 두 런에서 입력이 **동일**해야 한다.
거기서 코사인이 1.0 이 아니면 정렬이 깨진 것이므로 나머지 수치도 못 믿는다.
"""
import os, sys
import numpy as np

S = '/private/tmp/claude-501/-Users-angigyeom-Desktop-DEV-optillama/525ff5d7-3e78-403d-9ac0-15e30e1a2ad7/scratchpad'
N = 3072
L = 48
WL = [0] + [1 + ((il - 1) // 3) * 3 for il in range(1, L)]
for v in (45, 46, 47):
    WL[v] = v


def load(d, il, n):
    p = f'{S}/{d}/ffn_norm-{il}_(reshaped).f32'
    if not os.path.exists(p):
        return None
    a = np.fromfile(p, dtype=np.float32)
    a = a[:(a.size // N) * N].reshape(-1, N)
    return a[:n] if a.shape[0] >= n else None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    rows = []
    for il in range(1, L):
        A, B = load('a_unf', il, n), load('a_fold', il, n)
        if A is None or B is None:
            continue
        rows.append((il, WL[il] != il,
                     float(np.linalg.norm(B - A) / np.linalg.norm(A)),
                     float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B))),
                     float(np.linalg.norm(B) / np.linalg.norm(A))))

    # --- 정렬 검증: 접히지 않은 층은 동일해야 한다 ---
    # 진짜 불변량: 첫 접기(층 2) 이전 층만 두 런에서 동일하다. 층 4 는 접히지 않아도
    # 입력이 층 2·3 의 접기를 거쳐 오므로 달라지는 것이 정상이다.
    unf = [r for r in rows if r[0] < 2]
    bad = [r for r in unf if r[3] < 0.9999]
    print(f'[정렬검증] 첫 접기 이전 층 {len(unf)}개 중 코사인<0.9999: {len(bad)}')
    if bad:
        for r in bad[:5]:
            print(f'   층 {r[0]}: 코사인 {r[3]:.4f} — 정렬 깨짐')
        print('   !! 아래 수치 신뢰 불가')
        return
    print('   PASS — 정렬 정상\n')

    print(f"{'층':>3} {'wl':>3} {'상대차':>7} {'코사인':>7} {'노름비':>7}")
    for il, f, d, cs, nr in rows:
        if f:
            print(f'{il:3d} {WL[il]:3d} {d:7.3f} {cs:7.3f} {nr:7.3f}')

    fold = [r for r in rows if r[1]]
    if fold:
        ds = np.array([r[2] for r in fold]); cs = np.array([r[3] for r in fold])
        print(f'\n접힌 층 {len(fold)}개: 상대차 중앙 {np.median(ds):.3f} (최대 {ds.max():.3f}) '
              f'| 코사인 중앙 {np.median(cs):.3f} (최소 {cs.min():.3f})')
        big = np.median(ds) > 0.3 or np.median(cs) < 0.9
        print('판정:', '드리프트 큼 → 목적함수를 propagated trajectory 로 교체 가능'
              if big else '드리프트 작음 → teacher forcing 은 원인 아님. 사다리 종료')


if __name__ == '__main__':
    main()
