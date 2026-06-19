"""
NOMS-LAB connectome-gen : 단계 V-생성 — 초파리 유충 connectome 생성+구조검증

gpu_pipeline 모델을 학습 → 초파리 connectome을 생성(전수 스코어링) →
구조통계(밀도·상호성·허브·세포타입간 흐름)를 실제 vs 생성 vs 무작위 비교.
벌레에서 한 구조검증을 다른 종(곤충 뇌)에서 재현 = 생성 일반성.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_pipeline import ScalableEdgeGen, sample_neg, NEG_RATIO, BATCH

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "data", "processed", "pipeline_fly_larva.npz")
OUT = r"D:\NOMS-LAB-D\connectome-gen\outputs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


@torch.no_grad()
def full_logits(model, N, type_oh, pos, dscale, chunk=300):
    L = torch.empty(N, N, device=DEV)
    allj = torch.arange(N, device=DEV)
    for s0 in range(0, N, chunk):
        idx = torch.arange(s0, min(s0 + chunk, N), device=DEV)
        src = idx.repeat_interleave(N); dst = allj.repeat(len(idx))
        L[idx] = model(src, dst, type_oh, pos, dscale).view(len(idx), N)
    return L


def stats(A, ntype, T):
    k = A.shape[0]; n = A.sum()
    out = A.sum(1); inn = A.sum(0)
    M = torch.zeros(T, T, device=DEV)
    s, dst = A.nonzero(as_tuple=True)
    M.index_put_((ntype[s], ntype[dst]), torch.ones(len(s), device=DEV), accumulate=True)
    M = (M / M.sum()).flatten()
    return dict(density=float(n / (k * (k - 1))), reciprocity=float((A * A.T).sum() / max(n, 1)),
                mean_deg=float(out.mean()), max_out=int(out.max()), max_in=int(inn.max())), M


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-8))


def main():
    d = np.load(IN, allow_pickle=True)
    N = int(d["num_nodes"]); edges = d["edges"]; ntype_np = d["node_type"]; pos = d["pos"]
    T = int(ntype_np.max()) + 1
    ntype = torch.tensor(ntype_np, device=DEV)
    type_oh = torch.eye(T, device=DEV)[ntype]
    pos_t = torch.tensor(pos, dtype=torch.float32, device=DEV)
    E = torch.tensor(edges, dtype=torch.long, device=DEV)
    dscale = torch.tensor(1.0, device=DEV)
    print(f"초파리 유충: {N}뉴런 {len(E):,}엣지 {T}타입")

    # 학습 (전 엣지 양성 + negative sampling)
    model = ScalableEdgeGen(N, T).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    steps = max(1, len(E) // BATCH)
    for ep in range(40):
        order = torch.randperm(len(E), device=DEV)
        for st in range(steps):
            b = order[st * BATCH:(st + 1) * BATCH]
            ps, pd = E[b, 0], E[b, 1]
            ns, nd = sample_neg(len(b) * NEG_RATIO, N, None, DEV)
            src = torch.cat([ps, ns]); dst = torch.cat([pd, nd])
            y = torch.cat([torch.ones(len(ps), device=DEV), torch.zeros(len(ns), device=DEV)])
            opt.zero_grad(); lossf(model(src, dst, type_oh, pos_t, dscale), y).backward(); opt.step()

    # 생성: 전수 스코어 → 밀도맞춤 → 샘플
    L = full_logits(model, N, type_oh, pos_t, dscale)
    eye = torch.eye(N, dtype=torch.bool, device=DEV); L[eye] = -50.0
    A_real = torch.zeros(N, N, device=DEV); A_real[E[:, 0], E[:, 1]] = 1.0; A_real[eye] = 0
    target = A_real.sum().item()
    lo, hi = -20.0, 20.0
    for _ in range(60):
        c = (lo + hi) / 2
        if torch.sigmoid(L + c).sum().item() > target: hi = c
        else: lo = c
    pg = torch.sigmoid(L + c); pg[eye] = 0
    A_gen = (torch.rand(N, N, device=DEV) < pg).float(); A_gen[eye] = 0
    dens = target / (N * (N - 1))
    A_rnd = (torch.rand(N, N, device=DEV) < dens).float(); A_rnd[eye] = 0

    sr, Mr = stats(A_real, ntype, T); sg, Mg = stats(A_gen, ntype, T); sn, Mn = stats(A_rnd, ntype, T)
    print(f"\n=== 초파리 생성 vs 실제 (구조) ===")
    print(f"{'지표':<12}{'실제':>10}{'생성':>10}{'무작위':>10}")
    for key in ["density", "reciprocity", "mean_deg", "max_out", "max_in"]:
        f = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
        print(f"{key:<12}{f(sr[key]):>10}{f(sg[key]):>10}{f(sn[key]):>10}")
    print(f"\n세포타입간 흐름(18x18) 실제와 상관:  생성 {corr(Mg, Mr):.3f}   무작위 {corr(Mn, Mr):.3f}")
    np.savez_compressed(os.path.join(OUT, "phase5_fly_gen.npz"), prob=pg.cpu().numpy())
    print(f"저장: {os.path.join(OUT, 'phase5_fly_gen.npz')}")


if __name__ == "__main__":
    main()
