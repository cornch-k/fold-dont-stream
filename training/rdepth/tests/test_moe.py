import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import GPT, GPTConfig, CONFIGS

def _mk(name):
    torch.manual_seed(0)
    return GPT(CONFIGS[name])

def test_dense_regression():
    # MoE 추가가 dense 경로를 바꾸지 않았는지 — 1차 사이클과 동일 파라미터 수
    small = _mk("small")
    per_layer = 4 * 512 * 512 + 3 * 512 * 1408
    emb = 4096 * 512 + 512 * 512
    assert abs(small.num_unique_params() - (4 * per_layer + emb + 9 * 512)) < 5000

def test_moe_param_counts():
    ms, mlr, mlg = _mk("moe-small"), _mk("moe-loop-rr"), _mk("moe-large")
    # loop는 small + 루프임베딩(3×384)만 초과
    assert 0 < mlr.num_unique_params() - ms.num_unique_params() < 2000
    # large는 MoE층 4개분 이상 큼
    per_moe_layer = 16 * 3 * 384 * 512 + 384 * 16 + 4 * 384 * 384
    assert mlg.num_unique_params() - ms.num_unique_params() > 3.9 * per_moe_layer

def test_sticky_touch_equals_topk():
    torch.manual_seed(0)
    m = GPT(CONFIGS["moe-smoke"])  # sticky, R=3
    m.eval()
    m.reset_expert_stats(True)
    x = torch.randint(0, 4096, (2, 64))
    with torch.no_grad():
        m(x)
    assert abs(m.expert_stats() - m.cfg.top_k) < 1e-9  # sticky: 루프 합산 접촉 = top_k

def test_rr_touch_exceeds_topk():
    torch.manual_seed(0)
    cfg = GPTConfig(d=128, n_head=4, use_moe=True, n_experts=4, expert_hidden=64, ctx=128,
                    n_pre=1, n_loop_layers=2, n_loops=3, n_post=1, route_policy="rr")
    m = GPT(cfg)
    m.eval()
    m.reset_expert_stats(True)
    x = torch.randint(0, 4096, (2, 64))
    with torch.no_grad():
        m(x)
    assert m.expert_stats() > cfg.top_k  # 재라우팅: 루프마다 다른 expert 접촉

def test_moe_causality():
    torch.manual_seed(0)
    m = GPT(CONFIGS["moe-smoke"])
    m.eval()
    x = torch.randint(0, 4096, (1, 32))
    y1, _ = m(x)
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 4096
    y2, _ = m(x2)
    assert torch.allclose(y1[0, :31], y2[0, :31], atol=1e-5)

def test_moe_aux_and_backward():
    m = _mk("moe-loop-rr")
    x = torch.randint(0, 4096, (2, 64))
    _, loss = m(x, x)
    assert m.last_aux is not None and torch.isfinite(m.last_aux) and m.last_aux.item() > 0
    (loss + 0.01 * m.last_aux).backward()
    blk = m.loop_block[0]
    assert blk.moe.router.weight.grad is not None
    assert blk.moe.w1.grad is not None

def test_dense_sparse_equivalence():
    torch.manual_seed(0)
    m = GPT(CONFIGS["moe-smoke"])
    m.eval()
    x = torch.randint(0, 4096, (2, 64))
    with torch.no_grad():
        y_dense, _ = m(x)
        for mod in m.modules():
            if hasattr(mod, "dense_compute"):
                mod.dense_compute = False
        y_sparse, _ = m(x)
    assert torch.allclose(y_dense, y_sparse, atol=1e-4)

def test_resume_roundtrip(tmp_path):
    # 체크포인트 저장→CPU 로드→gen 복원까지 — CUDA 재개 버그 회귀 방지
    import numpy as np
    torch.manual_seed(0)
    m = GPT(CONFIGS["moe-smoke"])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(1337)
    gen.manual_seed(42)
    p = str(tmp_path / "ck.pt")
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": 7, "gen": gen.get_state()}, p)
    st = torch.load(p, map_location="cpu", weights_only=False)
    m2 = GPT(CONFIGS["moe-smoke"])
    m2.load_state_dict(st["model"])
    gen2 = torch.Generator()
    gen2.set_state(st["gen"])  # 여기서 TypeError 나면 재개 버그 재발
    assert st["step"] == 7
    assert torch.randint(0, 100, (3,), generator=gen2).tolist() == torch.randint(0, 100, (3,), generator=gen).tolist()
