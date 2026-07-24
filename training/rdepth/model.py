"""순환 깊이 GPT. pre → (loop block × R) → post. middle-cycle 공유."""
import dataclasses
import torch, torch.nn as nn, torch.nn.functional as F

@dataclasses.dataclass
class GPTConfig:
    vocab_size: int = 4096
    ctx: int = 512
    d: int = 512
    n_head: int = 8
    ffn_hidden: int = 1408
    n_pre: int = 1
    n_loop_layers: int = 2
    n_loops: int = 1
    n_post: int = 1
    inject_input: bool = True
    use_loop_emb: bool = True
    use_moe: bool = False
    n_experts: int = 16
    top_k: int = 2
    expert_hidden: int = 512
    route_policy: str = "rr"  # "rr"=루프마다 재라우팅 | "sticky"=첫 루프 선택 재사용

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

class MoEFFN(nn.Module):
    """top-k 소프트맥스 라우팅 MoE. cache/reuse로 루프 간 sticky 라우팅 지원."""
    def __init__(self, cfg):
        super().__init__()
        self.n_e, self.k = cfg.n_experts, cfg.top_k
        self.router = nn.Linear(cfg.d, cfg.n_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d, cfg.expert_hidden))
        self.w3 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d, cfg.expert_hidden))
        self.w2 = nn.Parameter(torch.empty(cfg.n_experts, cfg.expert_hidden, cfg.d))
        for p in (self.w1, self.w3, self.w2):
            nn.init.normal_(p, std=0.02)
        self._aux = None    # 이번 forward의 load-balance 손실 (GPT가 수거, 루프 누적)
        self._touch = None  # 평가용 (N, n_e) bool — 루프 누적 expert 접촉
        self.dense_compute = True  # MPS: einsum 밀집 경로 (CUDA 희소 디스패치는 False로)
    def forward(self, x, cache=None, reuse=False):
        B, T, C = x.shape
        h = x.view(-1, C)
        if reuse and cache is not None and id(self) in cache:
            gate, idx = cache[id(self)]
        else:
            probs = F.softmax(self.router(h), dim=-1)
            gate, idx = probs.topk(self.k, dim=-1)
            gate = gate / gate.sum(-1, keepdim=True)
            f = torch.zeros(self.n_e, device=h.device).index_add_(
                0, idx.reshape(-1), torch.ones(idx.numel(), device=h.device)) / idx.numel()
            aux = self.n_e * (f * probs.mean(0)).sum()
            self._aux = aux if self._aux is None else self._aux + aux
            if cache is not None:
                cache[id(self)] = (gate, idx)
        if self._touch is not None:
            self._touch.scatter_(1, idx, True)
        if self.dense_compute:
            # MPS 친화 경로: 전 expert 밀집 계산 + top-k 게이트 후적용 (수학 동일, 동기화 0)
            ye = torch.einsum('neh,ehd->ned', F.silu(torch.einsum('nd,edh->neh', h, self.w1))
                              * torch.einsum('nd,edh->neh', h, self.w3), self.w2)
            gf = torch.zeros(h.size(0), self.n_e, device=h.device, dtype=ye.dtype)
            gf.scatter_(1, idx, gate.to(ye.dtype))
            return (ye * gf.unsqueeze(-1)).sum(1).view(B, T, C)
        out = torch.zeros_like(h)
        for e in range(self.n_e):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            he = h[tok]
            ye = (F.silu(he @ self.w1[e]) * (he @ self.w3[e])) @ self.w2[e]
            out[tok] += gate[tok, slot].unsqueeze(-1) * ye
        return out.view(B, T, C)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.d = cfg.n_head, cfg.d
        self.ln1, self.ln2 = RMSNorm(cfg.d), RMSNorm(cfg.d)
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.proj = nn.Linear(cfg.d, cfg.d, bias=False)
        if cfg.use_moe:
            self.moe = MoEFFN(cfg)
        else:
            self.w1 = nn.Linear(cfg.d, cfg.ffn_hidden, bias=False)
            self.w3 = nn.Linear(cfg.d, cfg.ffn_hidden, bias=False)
            self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d, bias=False)
    def forward(self, x, cache=None, reuse=False):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(B, T, C))
        h = self.ln2(x)
        if hasattr(self, "moe"):
            return x + self.moe(h, cache=cache, reuse=reuse)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d)
        self.pos_emb = nn.Embedding(cfg.ctx, cfg.d)
        self.pre = nn.ModuleList([Block(cfg) for _ in range(cfg.n_pre)])
        self.loop_block = nn.ModuleList([Block(cfg) for _ in range(cfg.n_loop_layers)])
        self.post = nn.ModuleList([Block(cfg) for _ in range(cfg.n_post)])
        self.loop_emb = nn.Embedding(cfg.n_loops, cfg.d) if (cfg.use_loop_emb and cfg.n_loops > 1) else None
        self.ln_f = RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied
        self.last_aux = None
        self._count_experts = False
        self._exp_unique = 0
        self._exp_slots = 0
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        h0 = x
        if self.cfg.use_moe:
            for m in self.modules():
                if isinstance(m, MoEFFN):
                    m._aux = None
                    m._touch = (torch.zeros(B * T, m.n_e, dtype=torch.bool, device=idx.device)
                                if self._count_experts else None)
        cache = {} if (self.cfg.use_moe and self.cfg.route_policy == "sticky") else None
        for b in self.pre:
            x = b(x)
        for r in range(self.cfg.n_loops):
            if self.cfg.inject_input and self.cfg.n_loops > 1:
                x = x + h0
            if self.loop_emb is not None:
                x = x + self.loop_emb.weight[r]
            for b in self.loop_block:
                x = b(x, cache=cache, reuse=(r > 0))
        for b in self.post:
            x = b(x)
        if self.cfg.use_moe:
            auxes = [m._aux for m in self.modules() if isinstance(m, MoEFFN) and m._aux is not None]
            self.last_aux = torch.stack(auxes).sum() if auxes else None
            if self._count_experts:
                for m in self.modules():
                    if isinstance(m, MoEFFN) and m._touch is not None:
                        self._exp_unique += int(m._touch.sum().item())
                        self._exp_slots += m._touch.shape[0]
                        m._touch = None
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
    def num_unique_params(self):
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                n += p.numel()
        return n
    def resident_bytes(self, dtype_bytes=2):
        return self.num_unique_params() * dtype_bytes
    def reset_expert_stats(self, enable=True):
        self._count_experts = enable
        self._exp_unique = 0
        self._exp_slots = 0
    def expert_stats(self):
        """(토큰, MoE층) 슬롯당 평균 유니크 expert 접촉 수 (루프 합산)."""
        return (self._exp_unique / self._exp_slots) if self._exp_slots else float("nan")

CONFIGS = {
    "small": GPTConfig(n_pre=1, n_loop_layers=2, n_loops=1, n_post=1),
    "loop":  GPTConfig(n_pre=1, n_loop_layers=2, n_loops=3, n_post=1),
    "large": GPTConfig(n_pre=1, n_loop_layers=6, n_loops=1, n_post=1),
    "smoke": GPTConfig(d=256, n_head=4, ffn_hidden=704, n_pre=1, n_loop_layers=2, n_loops=2, n_post=1, ctx=256),
    # R 스윕 (dense): 회복률-R 곡선 — small/loop(R=3)/large와 동일 하이퍼, R만 변경
    "loop-r2": GPTConfig(n_pre=1, n_loop_layers=2, n_loops=2, n_post=1),
    "loop-r4": GPTConfig(n_pre=1, n_loop_layers=2, n_loops=4, n_post=1),
    "loop-r6": GPTConfig(n_pre=1, n_loop_layers=2, n_loops=6, n_post=1),
    # 2차 사이클: 루프×MoE (docs/plans/2026-07-23-rdepth-moe-cycle2.md)
    "moe-small":   GPTConfig(d=384, n_head=6, use_moe=True, n_pre=1, n_loop_layers=2, n_loops=1, n_post=1),
    "moe-loop-rr": GPTConfig(d=384, n_head=6, use_moe=True, route_policy="rr", n_pre=1, n_loop_layers=2, n_loops=3, n_post=1),
    "moe-loop-st": GPTConfig(d=384, n_head=6, use_moe=True, route_policy="sticky", n_pre=1, n_loop_layers=2, n_loops=3, n_post=1),
    "moe-large":   GPTConfig(d=384, n_head=6, use_moe=True, n_pre=1, n_loop_layers=6, n_loops=1, n_post=1),
    "moe-smoke":   GPTConfig(d=128, n_head=4, use_moe=True, n_experts=4, expert_hidden=64, ctx=128,
                             n_pre=1, n_loop_layers=2, n_loops=3, n_post=1, route_policy="sticky"),
}
