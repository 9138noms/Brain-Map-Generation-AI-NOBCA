"""
NOMS-LAB connectome-gen : 단계 XIV — 정밀 구조 정복 (허브 + 상호성)

프로젝트 내내 못 잡던 두 가지를 직접 공략 (쥐 피질):
  - 허브: 임베딩 차원↑(32) + 학습↑ → 무거운 꼬리(heavy-tail) 차수분포
  - 상호성: 독립 샘플링은 i↔j를 못 만듦 → 상호성 교정(reverse 엣지 주입, 밀도 유지)
phase9 baseline(허브 407 vs 785, 상호성 0.13 vs 0.27) 대비 개선되나.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_pipeline import sample_neg, NEG_RATIO, BATCH

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
MOUSE = os.path.join(HERE, "..", "data", "processed", "pipeline_microns_mouse.npz")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EMB = 32
torch.manual_seed(0); np.random.seed(0)


class FineGen(nn.Module):
    def __init__(self, N, T):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * T + 1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.u = nn.Embedding(N, EMB); self.v = nn.Embedding(N, EMB); self.s = nn.Embedding(N, EMB)
        for e in (self.u, self.v, self.s): nn.init.normal_(e.weight, std=0.1)

    def forward(self, src, dst, toh, pos, dscale):
        dist = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True) / dscale
        base = self.mlp(torch.cat([toh[src], toh[dst], dist], -1)).squeeze(-1)
        return base + (self.u(src) * self.v(dst)).sum(-1) + (self.s(src) * self.s(dst)).sum(-1)


def recip(A):
    n = A.sum()
    return float((A * A.t()).sum() / n.clamp(min=1))


def stats(A):
    out = A.sum(1)
    return dict(max_out=int(out.max()), deg_std=float(out.std()), recip=recip(A),
                density=float(A.sum() / (A.shape[0] * (A.shape[0] - 1))))


def main():
    d = np.load(MOUSE, allow_pickle=True)
    N = int(d["num_nodes"]); E = d["edges"]; nt = d["node_type"]; pos = d["pos"]
    T = int(nt.max()) + 1
    toh = torch.eye(T, device=DEV)[torch.tensor(nt, device=DEV)]
    P = torch.tensor(pos, dtype=torch.float32, device=DEV)
    Et = torch.tensor(E, dtype=torch.long, device=DEV)
    dscale = (P[Et[:, 0]] - P[Et[:, 1]]).norm(dim=-1).mean().clamp(min=1.0)
    eye = torch.eye(N, dtype=torch.bool, device=DEV)
    A_real = torch.zeros(N, N, device=DEV); A_real[Et[:, 0], Et[:, 1]] = 1; A_real[eye] = 0
    sr = stats(A_real)
    print(f"실제 쥐: 허브 {sr['max_out']}, 차수std {sr['deg_std']:.1f}, 상호성 {sr['recip']:.3f}")

    model = FineGen(N, T).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    lf = nn.BCEWithLogitsLoss(); steps = max(1, len(E) // BATCH)
    for ep in range(150):
        order = torch.randperm(len(E), device=DEV)
        for st in range(steps):
            b = order[st * BATCH:(st + 1) * BATCH]
            ns, nd = sample_neg(len(b) * NEG_RATIO, N, None, DEV)
            src = torch.cat([Et[b, 0], ns]); dst = torch.cat([Et[b, 1], nd])
            y = torch.cat([torch.ones(len(b), device=DEV), torch.zeros(len(ns), device=DEV)])
            opt.zero_grad(); lf(model(src, dst, toh, P, dscale), y).backward(); opt.step()

    # 전수 로짓 → 밀도맞춤
    with torch.no_grad():
        L = torch.empty(N, N, device=DEV); allj = torch.arange(N, device=DEV)
        for s0 in range(0, N, 400):
            idx = torch.arange(s0, min(s0 + 400, N), device=DEV)
            L[idx] = model(idx.repeat_interleave(N), allj.repeat(len(idx)), toh, P, dscale).view(len(idx), N)
    L[eye] = -50.0
    target = A_real.sum().item(); lo, hi = -20.0, 20.0
    for _ in range(60):
        c = (lo + hi) / 2
        if torch.sigmoid(L + c).sum().item() > target: hi = c
        else: lo = c
    pg = torch.sigmoid(L + c); pg[eye] = 0
    A0 = (torch.rand(N, N, device=DEV) < pg).float(); A0[eye] = 0
    s0 = stats(A0)
    print(f"\n[baseline 독립샘플] 허브 {s0['max_out']}, 차수std {s0['deg_std']:.1f}, 상호성 {s0['recip']:.3f}")

    # 차수 교정: 각 노드가 실제 out-degree만큼 연결 (행별 바이어스) → 허브 보장
    real_out = A_real.sum(1)
    loc = torch.full((N,), -25.0, device=DEV); hic = torch.full((N,), 25.0, device=DEV)
    for _ in range(50):
        mid = (loc + hic) / 2
        over = torch.sigmoid(L + mid[:, None]).sum(1) > real_out
        hic = torch.where(over, mid, hic); loc = torch.where(over, loc, mid)
    pdc = torch.sigmoid(L + ((loc + hic) / 2)[:, None]); pdc[eye] = 0
    A_dc = (torch.rand(N, N, device=DEV) < pdc).float(); A_dc[eye] = 0
    sdc = stats(A_dc)
    print(f"[차수교정]         허브 {sdc['max_out']}, 차수std {sdc['deg_std']:.1f}, 상호성 {sdc['recip']:.3f}")

    # 상호성 교정: 차수교정본에 reverse 주입, 같은수 무작위 제거(밀도/차수 유지)
    A = A_dc.clone()
    target_r = sr["recip"]
    for it in range(40):
        cur = recip(A)
        if cur >= target_r - 0.005: break
        nonrec = (A > 0) & (A.t() == 0)             # i→j, j→i 없음
        ni, nj = nonrec.nonzero(as_tuple=True)
        k = max(1, len(ni) // 12)
        pick = torch.randperm(len(ni), device=DEV)[:k]
        A[nj[pick], ni[pick]] = 1.0                  # reverse 주입
        ei, ej = (A > 0).nonzero(as_tuple=True)      # 무작위 제거(밀도유지)
        rem = torch.randperm(len(ei), device=DEV)[:k]
        A[ei[rem], ej[rem]] = 0.0
    sc = stats(A)
    print(f"[상호성 교정 후]   허브 {sc['max_out']}, 차수std {sc['deg_std']:.1f}, 상호성 {sc['recip']:.3f}")

    print(f"\n=== 정밀구조 정복 결과 (쥐) ===")
    print(f"{'지표':<10}{'실제':>9}{'phase9':>9}{'독립':>9}{'차수교정':>9}{'+상호성':>9}")
    print(f"{'허브(max)':<10}{sr['max_out']:>9}{'407':>9}{s0['max_out']:>9}{sdc['max_out']:>9}{sc['max_out']:>9}")
    print(f"{'차수std':<10}{sr['deg_std']:>9.1f}{'75.9':>9}{s0['deg_std']:>9.1f}{sdc['deg_std']:>9.1f}{sc['deg_std']:>9.1f}")
    print(f"{'상호성':<10}{sr['recip']:>9.3f}{'0.125':>9}{s0['recip']:>9.3f}{sdc['recip']:>9.3f}{sc['recip']:>9.3f}")


if __name__ == "__main__":
    main()
