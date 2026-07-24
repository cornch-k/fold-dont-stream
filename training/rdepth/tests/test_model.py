import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import GPT, GPTConfig, CONFIGS

def _mk(name):
    torch.manual_seed(0)
    return GPT(CONFIGS[name])

def test_param_counts():
    small, loop, large = _mk("small"), _mk("loop"), _mk("large")
    per_layer = 4 * 512 * 512 + 3 * 512 * 1408
    emb = 4096 * 512 + 512 * 512  # tok(tied) + pos
    rms = 512
    assert abs(small.num_unique_params() - (4 * per_layer + emb + 9 * rms)) < 5000
    # loop는 small + loop_emb(3×512)만 초과
    assert 0 < loop.num_unique_params() - small.num_unique_params() < 2000
    assert large.num_unique_params() - small.num_unique_params() > 4 * per_layer * 0.99

def test_weight_sharing_and_depth():
    loop = _mk("loop")
    blocks = list(loop.pre) + list(loop.loop_block) + list(loop.post)
    assert len(blocks) == 4  # 유효 깊이 8이지만 유일 Block 4개

def test_loop1_equals_plain():
    torch.manual_seed(0)
    m = GPT(CONFIGS["small"])
    x = torch.randint(0, 4096, (2, 64))
    logits, _ = m(x)
    assert logits.shape == (2, 64, 4096)
    assert m.loop_emb is None  # n_loops=1이면 루프 임베딩 비활성

def test_causality():
    torch.manual_seed(0)
    m = GPT(CONFIGS["smoke"])
    m.eval()
    x = torch.randint(0, 4096, (1, 32))
    y1, _ = m(x)
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 4096
    y2, _ = m(x2)
    assert torch.allclose(y1[0, :31], y2[0, :31], atol=1e-5)

def test_loss_backward():
    m = _mk("loop")
    x = torch.randint(0, 4096, (2, 64))
    _, loss = m(x, x)
    loss.backward()
    assert loss.item() > 0
    assert m.loop_block[0].qkv.weight.grad is not None

def test_r_sweep_configs():
    # R 스윕 구성: 유일 파라미터가 small과 루프 임베딩(R×512)만 차이나야 함
    base = _mk("small").num_unique_params()
    for name, r in [("loop-r2", 2), ("loop-r4", 4), ("loop-r6", 6)]:
        m = _mk(name)
        assert m.num_unique_params() - base == r * 512, name
        x = torch.randint(0, 4096, (2, 32))
        logits, loss = m(x, x)
        assert logits.shape == (2, 32, 4096) and loss.item() > 0, name
