#!/usr/bin/env python3
"""접힌 층의 논리층별 보정 파라미터를 층별 국소 회귀로 맞춘다.

핵심: FFN 블록은 자기 입력의 순수 함수다. 무접기 모델에서 층 il 의 FFN 입력 X 만
덤프해두면, 목표 출력 Y = FFN_il(X) 는 층 il 자신의 가중치로 오프라인 재계산할 수 있다.
118B 전체는 학습 그래프에 들어가지 않는다.

사다리 1단계: γ_il = argmin ‖γ·moe_out_wl(X) + shexp_il(X) − Y‖  (닫힌형 1-D 최소제곱)
"""
import sys, os, glob, json, argparse
import numpy as np

sys.path.insert(0, '/Users/angigyeom/Desktop/DEV/optillama/llama.cpp/gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize

MODEL_GLOB = '/Users/angigyeom/Desktop/DEV/optillama/models/Laguna-S-2.1-GGUF/UD-IQ4_XS/*.gguf'
ACTS = '/private/tmp/claude-501/-Users-angigyeom-Desktop-DEV-optillama/525ff5d7-3e78-403d-9ac0-15e30e1a2ad7/scratchpad/acts'

N_EMBD, N_EXPERT, N_USED, N_FF = 3072, 256, 10, 1024
W_SCALE = 2.5          # laguna.expert_weights_scale
NORM_W  = True         # laguna.expert_weights_norm


# ---------------------------------------------------------------- 텐서 접근
class Model:
    """샤드 전체의 텐서를 이름으로 조회. 역양자화는 요청 시에만, 캐시 없음(메모리)."""
    def __init__(self):
        self.t = {}
        for p in sorted(glob.glob(MODEL_GLOB)):
            r = GGUFReader(p)
            for x in r.tensors:
                self.t[x.name] = x
        if not self.t:
            raise SystemExit('GGUF 텐서를 못 찾음')

    def raw(self, name):
        return self.t[name]

    def deq(self, name):
        x = self.t[name]
        a = dequantize(x.data, x.tensor_type).astype(np.float32)
        return a.reshape(tuple(int(v) for v in reversed(x.shape)))

    def deq_expert(self, name, e):
        """3-D 전문가 스택에서 전문가 e 한 장만 역양자화. 스택 전체를 메모리에 올리지 않는다."""
        x = self.t[name]
        ne = [int(v) for v in x.shape]          # (n0, n1, n_expert)
        rows_per_expert = ne[1]
        blk = x.data.reshape(ne[2] * rows_per_expert, -1)
        sl = blk[e * rows_per_expert:(e + 1) * rows_per_expert]
        return dequantize(sl, x.tensor_type).astype(np.float32).reshape(rows_per_expert, ne[0])


# ---------------------------------------------------------------- MoE forward
def rms_norm(x, w, eps=1e-6):
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def route(m, il, X):
    """laguna 라우팅: sigmoid + score-correction bias -> top-k -> (선택) 합 정규화 -> scale."""
    Wg = m.deq(f'blk.{il}.ffn_gate_inp.weight')            # (n_expert, n_embd)
    logits = X @ Wg.T                                       # (T, n_expert)
    probs = 1.0 / (1.0 + np.exp(-logits))
    # 실제 GGUF 이름은 exp_probs_b.bias 다. 예전 이름은 절대 매치되지 않아
    # bias 가 조용히 0 으로 대체됐다 — 부록 A(10)과 같은 계열의 함정.
    bname = f'blk.{il}.exp_probs_b.bias'
    assert bname in m.t, f'{bname} 없음 — 라우팅 바이어스 경로가 검증되지 않았다'
    sel = probs + (m.deq(bname).reshape(-1) if bname in m.t else 0.0)
    idx = np.argpartition(-sel, N_USED - 1, axis=-1)[:, :N_USED]
    w = np.take_along_axis(probs, idx, axis=-1)             # 가중치는 bias 없는 probs
    if NORM_W:
        w = w / np.maximum(w.sum(-1, keepdims=True), 6.103515625e-5)
    return idx, (w * W_SCALE).astype(np.float32)


def moe_out(m, il, X, idx, w):
    """선택된 전문가만 스트리밍으로 역양자화해 누적. (T, n_embd)"""
    T = X.shape[0]
    out = np.zeros((T, N_EMBD), dtype=np.float32)
    for e in np.unique(idx):
        tok, slot = np.nonzero(idx == e)
        if tok.size == 0:
            continue
        g = m.deq_expert(f'blk.{il}.ffn_gate_exps.weight', e)   # (n_ff, n_embd)
        u = m.deq_expert(f'blk.{il}.ffn_up_exps.weight',   e)
        d = m.deq_expert(f'blk.{il}.ffn_down_exps.weight', e)   # (n_embd, n_ff)
        h = X[tok]
        a = h @ g.T
        a = a / (1.0 + np.exp(-a))                              # silu
        y = (a * (h @ u.T)) @ d.T
        np.add.at(out, tok, y * w[tok, slot][:, None])
    return out


def shexp_out(m, il, X):
    g = m.deq(f'blk.{il}.ffn_gate_shexp.weight')
    u = m.deq(f'blk.{il}.ffn_up_shexp.weight')
    d = m.deq(f'blk.{il}.ffn_down_shexp.weight')
    a = X @ g.T
    a = a / (1.0 + np.exp(-a))
    return (a * (X @ u.T)) @ d.T


def ffn_block(m, il, X):
    idx, w = route(m, il, X)
    return moe_out(m, il, X, idx, w) + shexp_out(m, il, X)


# ---------------------------------------------------------------- 활성값
def load_X(il, n=None):
    p = f'{ACTS}/blk.{il}.ffn_gate_exps.weight.f32'
    a = np.fromfile(p, dtype=np.float32).reshape(-1, N_EMBD)
    return a[:n] if n else a


# ---------------------------------------------------------------- 사다리 1단계
def fit_gamma(m, il, wl, X, shexp_logical=True):
    """γ_il = <moe_wl, Y − shexp> / <moe_wl, moe_wl>  (닫힌형)"""
    Y = ffn_block(m, il, X)                       # 원본 층 il 의 출력 = 목표
    idx, w = route(m, wl, X)                      # 접기: 라우터도 물리층
    Mo = moe_out(m, wl, X, idx, w)
    Sh = shexp_out(m, il if shexp_logical else wl, X)
    R = Y - Sh
    denom = float((Mo * Mo).sum())
    g = float((Mo * R).sum() / denom) if denom > 0 else 0.0
    def rel(pred): return float(np.linalg.norm(pred - Y) / np.linalg.norm(Y))
    return {
        'il': il, 'wl': wl, 'gamma': g,
        'rel_g1':   rel(Mo + Sh),
        'rel_g05':  rel(0.5 * Mo + Sh),
        'rel_gfit': rel(g * Mo + Sh),
        'rel_g0':   rel(Sh),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=512)
    ap.add_argument('--layers', type=str, default='')   # "2,3,5" 비우면 접힌 층 전부
    ap.add_argument('--out', type=str, default='')
    a = ap.parse_args()

    # 현재 최선 사상: R=3, lead=1, 층 45/46/47 고정
    L = 48
    wlmap = [0] + [1 + ((il - 1) // 3) * 3 for il in range(1, L)]
    for v in (45, 46, 47):
        wlmap[v] = v

    layers = [int(x) for x in a.layers.split(',') if x] or [il for il in range(1, L) if wlmap[il] != il]

    m = Model()

    # 배관 검증: il == wl 이면 γ*=1, 상대오차 0 이어야 한다. 라우팅 결정성·전문가
    # 스트리밍·누적이 일관되지 않으면 여기서 깨진다.
    chk = fit_gamma(m, 4, 4, load_X(4, 128))
    ok = abs(chk['gamma'] - 1.0) < 1e-4 and chk['rel_gfit'] < 1e-5
    print(f"[배관검증] il=wl=4 -> γ*={chk['gamma']:.6f} rel={chk['rel_gfit']:.2e}  {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print('  !! 하니스 내부 불일치. 아래 수치는 신뢰할 수 없음.', flush=True)
    res = []
    for il in layers:
        wl = wlmap[il]
        X = load_X(il, a.tokens)
        r = fit_gamma(m, il, wl, X)
        res.append(r)
        print(f"il={r['il']:2d} wl={r['wl']:2d} γ*={r['gamma']:+.3f} | "
              f"rel: γ=1 {r['rel_g1']:.3f}  γ=0.5 {r['rel_g05']:.3f}  "
              f"γ* {r['rel_gfit']:.3f}  γ=0 {r['rel_g0']:.3f}", flush=True)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
        print('저장:', a.out)


if __name__ == '__main__':
    main()
