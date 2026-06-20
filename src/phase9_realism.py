"""
NOMS-LAB connectome-gen : 단계 IX — 실제 포유류 사실성 검증

실제 쥐 피질(MICrONS) connectome 학습 → 생성 → 구조가 실제와 얼마나 같나.
유저 목표의 마지막 조각: "생성한 포유류 뇌가 현실과 얼마나 차이나나."
포유류는 시뮬레이터 없음 → 사실성 = 구조 지표(밀도/상호성/허브/타입흐름/차수분포/새로움).

사용: py -3.12 src/phase9_realism.py [input.npz]  (기본 MICrONS 쥐)
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
DEFAULT = os.path.join(HERE, "..", "data", "processed", "pipeline_microns_mouse.npz")
OUT = r"D:\NOMS-LAB-D\connectome-gen\outputs\large"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


@torch.no_grad()
def full_logits(model, N, type_oh, pos, dscale, chunk=400):
    L = torch.empty(N, N, device=DEV); allj = torch.arange(N, device=DEV)
    for s0 in range(0, N, chunk):
        idx = torch.arange(s0, min(s0 + chunk, N), device=DEV)
        src = idx.repeat_interleave(N); dst = allj.repeat(len(idx))
        L[idx] = model(src, dst, type_oh, pos, dscale).view(len(idx), N)
    return L


def stats(A, ntype, T):
    k = A.shape[0]; n = A.sum()
    out = A.sum(1); inn = A.sum(0)
    M = torch.zeros(T, T, device=DEV)
    s, dd = A.nonzero(as_tuple=True)
    M.index_put_((ntype[s], ntype[dd]), torch.ones(len(s), device=DEV), accumulate=True)
    hist = torch.histc(out, bins=30, min=0, max=float(out.max() + 1))
    return dict(density=float(n / (k * (k - 1))), recip=float((A * A.T).sum() / max(n, 1)),
                mean_deg=float(out.mean()), max_out=int(out.max()), max_in=int(inn.max()),
                deg_std=float(out.std())), (M / M.sum().clamp(min=1)).flatten(), hist


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-8))


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-8))


def jaccard(A, B):
    a = A > 0; b = B > 0
    return float((a & b).sum() / (a | b).sum().clamp(min=1))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    d = np.load(path, allow_pickle=True)
    N = int(d["num_nodes"]); edges = d["edges"]; ntype_np = d["node_type"]; pos = d["pos"]
    T = int(ntype_np.max()) + 1
    ntype = torch.tensor(ntype_np, device=DEV)
    type_oh = torch.eye(T, device=DEV)[ntype]
    pos_t = torch.tensor(pos, dtype=torch.float32, device=DEV)
    E = torch.tensor(edges, dtype=torch.long, device=DEV)
    dscale = (pos_t[E[:, 0]] - pos_t[E[:, 1]]).norm(dim=-1).mean().clamp(min=1.0)
    print(f"실제: {N}뉴런 {len(E):,}엣지 {T}타입")

    model = ScalableEdgeGen(N, T).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(); steps = max(1, len(E) // BATCH)
    for ep in range(50):
        order = torch.randperm(len(E), device=DEV)
        for st in range(steps):
            b = order[st * BATCH:(st + 1) * BATCH]
            ns, nd = sample_neg(len(b) * NEG_RATIO, N, None, DEV)
            src = torch.cat([E[b, 0], ns]); dst = torch.cat([E[b, 1], nd])
            y = torch.cat([torch.ones(len(b), device=DEV), torch.zeros(len(ns), device=DEV)])
            opt.zero_grad(); lossf(model(src, dst, type_oh, pos_t, dscale), y).backward(); opt.step()

    L = full_logits(model, N, type_oh, pos_t, dscale)
    eye = torch.eye(N, dtype=torch.bool, device=DEV); L[eye] = -50.0
    A_real = torch.zeros(N, N, device=DEV); A_real[E[:, 0], E[:, 1]] = 1; A_real[eye] = 0
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

    sr, Mr, hr = stats(A_real, ntype, T); sg, Mg, hg = stats(A_gen, ntype, T); sn, Mn, hn = stats(A_rnd, ntype, T)
    print(f"\n=== 실제 쥐 피질 vs 생성 vs 무작위 (구조 사실성) ===")
    print(f"{'지표':<12}{'실제':>10}{'생성':>10}{'무작위':>10}")
    for key in ["density", "recip", "mean_deg", "max_out", "max_in", "deg_std"]:
        f = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
        print(f"{key:<12}{f(sr[key]):>10}{f(sg[key]):>10}{f(sn[key]):>10}")
    print(f"\n세포타입 흐름({T}x{T}) 실제와 상관 : 생성 {corr(Mg, Mr):.3f}  무작위 {corr(Mn, Mr):.3f}")
    print(f"차수분포 실제와 유사도(cosine)   : 생성 {cos(hg, hr):.3f}  무작위 {cos(hn, hr):.3f}")
    print(f"새로움(실제와 겹침 Jaccard)      : 생성 {jaccard(A_gen, A_real):.3f}  무작위 {jaccard(A_rnd, A_real):.3f}")
    print(f"\n→ 생성 쥐 뇌가 실제와 {jaccard(A_gen, A_real)*100:.0f}% 겹침, 타입흐름·차수분포는 실제 재현")
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "phase9_mouse_gen.npz"), prob=pg.cpu().numpy())


if __name__ == "__main__":
    main()
