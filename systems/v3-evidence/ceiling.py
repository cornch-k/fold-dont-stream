#!/usr/bin/env python3
"""도달 가능성 천장: 라우팅을 완벽히 맞혔다면 접힌 층이 얼마나 원본에 가까워질 수 있는가.

층 il 을 물리층 wl 로 접었을 때, 라우팅 분기가 만들어야 하는 목표는
    R = Y - shexp_il(X),   Y = FFN_il(X)
이고, wl 의 전문가 집합으로 도달 가능한 것은 (expert_weights_norm=True 이므로)
    2.5 * conv{E^wl_1(x), ..., E^wl_256(x)}
이다. 여기서 제약 없는 최소제곱은 그 볼록껍질보다 넓으므로 **잔차의 하한**을 준다.

하한이 이미 크면 → wl 의 전문가 집합이 il 의 함수를 표현하지 못한다.
                   어떤 어댑터도, 어떤 라우터 재적합도, 어떤 학습도 이 사상에서는 못 고친다.
하한이 작으면    → 문제는 라우팅이다. 어댑터·라우터 학습이 통할 여지가 있다.

검증: il 자기 자신의 전문가로 풀면 잔차가 0 이어야 한다(참해가 실현 가능집합 안에 있다).
"""
import sys, time
import numpy as np
import moefit as M

S = '/private/tmp/claude-501/-Users-angigyeom-Desktop-DEV-optillama/525ff5d7-3e78-403d-9ac0-15e30e1a2ad7/scratchpad'
N_EMBD, N_EXPERT = 3072, 256


def load_folded_X(il, n):
    """접힌 모델의 실제 궤적에서 층 il 의 FFN 입력."""
    p = f'{S}/a_fold/ffn_norm-{il}_(reshaped).f32'
    a = np.fromfile(p, dtype=np.float32)
    a = a[:(a.size // N_EMBD) * N_EMBD].reshape(-1, N_EMBD)
    return a[:n]


def all_expert_outputs(m, layer, X):
    """(T, 256, n_embd) — 전 전문가의 출력. 전문가 한 장씩 스트리밍."""
    T = X.shape[0]
    E = np.empty((T, N_EXPERT, N_EMBD), dtype=np.float32)
    for e in range(N_EXPERT):
        g = m.deq_expert(f'blk.{layer}.ffn_gate_exps.weight', e)
        u = m.deq_expert(f'blk.{layer}.ffn_up_exps.weight',   e)
        d = m.deq_expert(f'blk.{layer}.ffn_down_exps.weight', e)
        a = X @ g.T
        a = a / (1.0 + np.exp(-a))
        E[:, e, :] = (a * (X @ u.T)) @ d.T
    return E


def oracle_resid(E, R):
    """토큰마다 제약 없는 최소제곱. 반환: 상대잔차 (하한)."""
    T = R.shape[0]
    num = 0.0
    for t in range(T):
        A = E[t].T                       # (n_embd, 256)
        sol, *_ = np.linalg.lstsq(A, R[t], rcond=None)
        num += float(np.sum((A @ sol - R[t]) ** 2))
    return float(np.sqrt(num) / np.linalg.norm(R))


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    ils = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [5, 24, 41]

    L = 48
    wl = [0] + [1 + ((i - 1) // 3) * 3 for i in range(1, L)]
    for v in (45, 46, 47):
        wl[v] = v

    m = M.Model()
    print(f'토큰 {T}개, 접힌 모델 궤적 기준\n')
    print(f"{'층':>3} {'wl':>3} {'현재접기':>9} {'천장(wl)':>9} {'검증(il)':>9}  {'해석'}")

    for il in ils:
        t0 = time.time()
        X = load_folded_X(il, T)
        Y = M.ffn_block(m, il, X)
        Sh = M.shexp_out(m, il, X)
        R = Y - Sh
        nR = np.linalg.norm(R)

        idx, w = M.route(m, wl[il], X)
        Mo = M.moe_out(m, wl[il], X, idx, w)
        now = float(np.linalg.norm(Mo - R) / nR)

        E_wl = all_expert_outputs(m, wl[il], X)
        ceil_wl = oracle_resid(E_wl, R)
        del E_wl

        E_il = all_expert_outputs(m, il, X)
        ctrl = oracle_resid(E_il, R)
        del E_il

        verdict = ('라우팅이 병목 — 학습 여지 있음' if ceil_wl < 0.3
                   else '표현력 부족 — 이 사상에서는 학습해도 한계' if ceil_wl > 0.6
                   else '중간')
        flag = '' if ctrl < 0.05 else '  !!검증실패'
        print(f'{il:3d} {wl[il]:3d} {now:9.3f} {ceil_wl:9.3f} {ctrl:9.3f}  {verdict}{flag}'
              f'   [{time.time()-t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
