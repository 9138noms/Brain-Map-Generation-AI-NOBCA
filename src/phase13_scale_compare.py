"""
NOMS-LAB connectome-gen : 단계 XIII — 큰 뇌 vs 작은 뇌 (차이 + 학습능력)

같은 쥐 피질 규칙으로 여러 크기(500~8000뉴런) 생성 →
  구조 차이: 밀도, 평균/최대 차수, 군집계수(clustering)
  학습능력: reservoir 선형기억용량 + 비선형 계산능력
크기 따라 어떻게 변하나 = "큰 뇌가 더 똑똑한가" 답.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
MOUSE = os.path.join(HERE, "..", "data", "processed", "pipeline_microns_mouse.npz")
VIZ = r"D:\NOMS-LAB-D\connectome-gen\outputs\large\viz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
# 시간샘플 > 뉴런수 여야 readout 회귀가 안정 (8000뉴런 → 10000스텝)
T_RES, WASH, KMEM, RHO, INSC, RIDGE = 10000, 1000, 20, 0.6, 0.3, 1e-3


class Rule(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * T + 1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, ti, tj, dist):
        return self.mlp(torch.cat([ti, tj, dist], -1)).squeeze(-1)


def train_rule(d):
    N = int(d["num_nodes"]); E = d["edges"]; nt = d["node_type"]; pos = d["pos"]
    T = int(nt.max()) + 1
    P = torch.tensor(pos, dtype=torch.float32, device=DEV)
    toh = torch.eye(T, device=DEV)[torch.tensor(nt, device=DEV)]
    Et = torch.tensor(E, dtype=torch.long, device=DEV)
    dscale = (P[Et[:, 0]] - P[Et[:, 1]]).norm(dim=-1).mean().clamp(min=1.0)
    model = Rule(T).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    lf = nn.BCEWithLogitsLoss()
    for ep in range(120):
        ps, pd = Et[:, 0], Et[:, 1]
        ns = torch.randint(0, N, (len(ps) * 5,), device=DEV); nd = torch.randint(0, N, (len(ps) * 5,), device=DEV)
        src = torch.cat([ps, ns]); dst = torch.cat([pd, nd])
        dist = (P[src] - P[dst]).norm(dim=-1, keepdim=True) / dscale
        y = torch.cat([torch.ones(len(ps), device=DEV), torch.zeros(len(ns), device=DEV)])
        opt.zero_grad(); lf(model(toh[src], toh[dst], dist), y).backward(); opt.step()
    span = (P.max(0).values - P.min(0).values).clamp(min=1.0)
    return model, T, float(dscale), N / span.prod().item(), np.bincount(nt, minlength=T) / N, len(E) / N


@torch.no_grad()
def gen_brain(model, T, N, dscale, dvol, tdist, mean_deg, C=96):
    side = (N / dvol) ** (1 / 3)
    pos = torch.rand(N, 3, device=DEV) * side
    types = torch.tensor(np.random.choice(T, N, p=tdist), device=DEV)
    EYE = torch.eye(T, device=DEV); toh = EYE[types]
    cs = (300 / dvol) ** (1 / 3); G = int(side / cs) + 2
    cell = (pos / cs).floor().long().clamp(0, G - 1)
    cid = cell[:, 0] * G * G + cell[:, 1] * G + cell[:, 2]
    order = torch.argsort(cid); uniq, inv, cnt = torch.unique(cid[order], return_inverse=True, return_counts=True)
    starts = torch.cumsum(cnt, 0) - cnt
    A = torch.zeros(N, N, device=DEV)
    for p0 in range(0, N, 20000):
        ps = torch.arange(p0, min(p0 + 20000, N), device=DEV)
        st = starts[inv[ps]]; ln = cnt[inv[ps]].clamp(min=1)
        off = (torch.rand(len(ps), C, device=DEV) * ln.unsqueeze(1)).long()
        cand = order[(st.unsqueeze(1) + off).reshape(-1)]
        si = order[ps].repeat_interleave(C)
        dist = (pos[si] - pos[cand]).norm(dim=-1, keepdim=True) / dscale
        p = torch.sigmoid(model(toh[si], toh[cand], dist)).view(len(ps), C)
        p = p * (mean_deg / C) / p.mean().clamp(min=1e-6)
        fire = torch.rand_like(p) < p
        rows, cols = fire.nonzero(as_tuple=True)
        A[order[ps][rows], cand.view(len(ps), C)[rows, cols]] = 1.0
    A.fill_diagonal_(0)
    return A


def clustering(A, sample=150):
    N = A.shape[0]; U = ((A + A.t()) > 0).float()
    idx = torch.randperm(N, device=DEV)[:sample]
    cs = []
    for i in idx.tolist():
        nb = torch.where(U[i] > 0)[0]
        k = len(nb)
        if k < 2: continue
        sub = U[nb][:, nb]
        cs.append((sub.sum() / (k * (k - 1))).item())
    return float(np.mean(cs)) if cs else 0.0


def capacity(A):
    N = A.shape[0]
    W = torch.randn(N, N, device=DEV) * A
    # 스펙트럼 반경 ≈ √(평균 입력차수) (iid 가중치) — 모든 크기서 안정적 스케일
    rho = (A.sum() / N).sqrt().clamp(min=1e-3)
    W = W * (RHO / rho)
    Win = (torch.rand(N, device=DEV) - 0.5) * 2 * INSC
    u = (torch.rand(T_RES, device=DEV) - 0.5) * 1.6
    X = torch.zeros(T_RES, N, device=DEV); x = torch.zeros(N, device=DEV)
    for t in range(T_RES):
        x = torch.tanh(W @ x + Win * u[t]); X[t] = x
    Xs = X[WASH:]; us = u[WASH:]
    Xb = torch.cat([Xs, torch.ones(len(Xs), 1, device=DEV)], 1)
    Ginv = torch.linalg.inv(Xb.t() @ Xb + RIDGE * torch.eye(Xb.shape[1], device=DEV))
    def r2(tg):
        pr = Xb @ (Ginv @ (Xb.t() @ tg))
        return float((1 - ((tg - pr) ** 2).sum() / ((tg - tg.mean()) ** 2).sum().clamp(min=1e-9)).clamp(0, 1))
    mc = sum(r2(torch.cat([torch.zeros(k, device=DEV), us[:-k]])) for k in range(1, KMEM + 1))
    nl = torch.zeros(len(Xs), device=DEV); nl[3:] = us[2:-1] * us[:-3]
    return mc, r2(nl)


def main():
    d = np.load(MOUSE, allow_pickle=True)
    model, T, dscale, dvol, tdist, mean_deg = train_rule(d)
    print(f"쥐 규칙 학습완료 (평균차수 {mean_deg:.0f})\n")
    scales = [500, 1000, 2000, 4000, 8000]
    rows = []
    print(f"{'뉴런':>7}{'밀도':>9}{'평균차수':>9}{'군집계수':>9}{'선형기억':>9}{'비선형':>8}")
    for N in scales:
        A = gen_brain(model, T, N, dscale, dvol, tdist, mean_deg)
        dens = A.sum().item() / (N * (N - 1)); md = A.sum(1).mean().item(); cl = clustering(A)
        mc, nl = capacity(A)
        rows.append((N, dens, md, cl, mc, nl))
        print(f"{N:>7,}{dens:>9.4f}{md:>9.1f}{cl:>9.3f}{mc:>9.2f}{nl:>8.3f}")

    rows = np.array(rows)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(rows[:, 0], rows[:, 1], "o-"); ax[0].set_title("Density vs size (sparser)"); ax[0].set_xlabel("neurons"); ax[0].set_xscale("log")
    ax[1].plot(rows[:, 0], rows[:, 3], "o-", color="tab:green"); ax[1].set_title("Clustering vs size"); ax[1].set_xlabel("neurons"); ax[1].set_xscale("log")
    ax[2].plot(rows[:, 0], rows[:, 4], "o-", color="tab:red", label="linear memory")
    ax[2].set_title("LEARNING CAPACITY vs size"); ax[2].set_xlabel("neurons"); ax[2].set_xscale("log"); ax[2].set_ylabel("memory capacity"); ax[2].legend()
    plt.tight_layout(); plt.savefig(f"{VIZ}/7_scale_compare.png", dpi=110); plt.close()
    print(f"\n시각화: {VIZ}/7_scale_compare.png")


if __name__ == "__main__":
    main()
