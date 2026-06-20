"""
NOMS-LAB connectome-gen : 단계 XVIII (Tier-A #3) — 발달 생성

Witvliet 8단계(출생→성체). 발달단계 조건부 생성 → 모델이 *성장 과정*을 배웠나.
단일 전역 임계값으로 각 단계 생성 → 엣지가 단계 따라 자연히 늘어나면(densification)
= "뇌가 자라는 과정"을 학습한 것. 실제 성장곡선 vs 생성 성장곡선.
"""
import os, sys
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
DEVN = os.path.join(HERE, "..", "data", "processed", "celegans_dev.npz")
VIZ = r"D:\NOMS-LAB-D\connectome-gen\outputs\large\viz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TIDX = {"sensory": 0, "inter": 1, "motor": 2, "modulatory": 3, "glia": 4, "other": 5}
EMB = 16; T = 6; NS = 8
torch.manual_seed(0); np.random.seed(0)


class DevGen(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * T + 2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.u = nn.Embedding(N, EMB); self.v = nn.Embedding(N, EMB); self.s = nn.Embedding(N, EMB)
        for e in (self.u, self.v, self.s): nn.init.normal_(e.weight, std=0.1)

    def forward(self, src, dst, toh, dist, stage):
        feat = torch.cat([toh[src], toh[dst], dist, stage], -1)
        return self.mlp(feat).squeeze(-1) + (self.u(src) * self.v(dst)).sum(-1) + (self.s(src) * self.s(dst)).sum(-1)


def main():
    d = np.load(DEVN, allow_pickle=True)
    chem = d["chem"]; pos = d["pos"]; present = d["present"]; types = d["node_type"]
    N = len(types)
    tix = np.array([TIDX.get(str(t), 5) for t in types])
    toh = torch.eye(T, device=DEV)[torch.tensor(tix, device=DEV)]

    # 단계별 (존재&좌표) 노드, 거리, 엣지
    st_idx, st_dist, st_edges, real_edges = [], [], [], []
    for s in range(NS):
        ok = present[s] & ~np.isnan(pos[s, :, 0]); idx = np.where(ok)[0]
        P = torch.tensor(pos[s, idx], dtype=torch.float32, device=DEV)
        st_idx.append(torch.tensor(idx, device=DEV))
        st_dist.append(torch.cdist(P, P))
        sub = chem[s][np.ix_(idx, idx)]; np.fill_diagonal(sub, 0)
        ei, ej = np.where(sub > 0)
        st_edges.append((torch.tensor(idx[ei], device=DEV), torch.tensor(idx[ej], device=DEV)))
        real_edges.append(len(ei))
    alld = torch.cat([dm[dm > 0] for dm in st_dist]).mean()
    st_dist = [dm / alld for dm in st_dist]
    # 글로벌 인덱스→단계내 위치 (거리 조회용)
    pos_in = [{int(v): k for k, v in enumerate(si.tolist())} for si in st_idx]

    model = DevGen(N).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(250):
        s = np.random.randint(NS)
        idx = st_idx[s]; k = len(idx); D = st_dist[s]
        pi, pj = st_edges[s]
        ne = len(pi) * 5
        ni = idx[torch.randint(0, k, (ne,), device=DEV)]; nj = idx[torch.randint(0, k, (ne,), device=DEV)]
        src = torch.cat([pi, ni]); dst = torch.cat([pj, nj])
        # 거리 조회 (단계내 위치)
        loc = torch.tensor([pos_in[s][int(x)] for x in src.tolist()], device=DEV)
        locd = torch.tensor([pos_in[s][int(x)] for x in dst.tolist()], device=DEV)
        dist = D[loc, locd].unsqueeze(-1)
        stage = torch.full((len(src), 1), s / (NS - 1), device=DEV)
        y = torch.cat([torch.ones(len(pi), device=DEV), torch.zeros(ne, device=DEV)])
        opt.zero_grad(); bce(model(src, dst, toh, dist, stage), y).backward(); opt.step()

    # 단일 전역 bias 로 각 단계 생성 → 엣지수 자연 증가하나
    @torch.no_grad()
    def stage_logits(s):
        idx = st_idx[s]; k = len(idx); D = st_dist[s]
        ii, jj = torch.meshgrid(torch.arange(k, device=DEV), torch.arange(k, device=DEV), indexing="ij")
        src = idx[ii.reshape(-1)]; dst = idx[jj.reshape(-1)]
        dist = D.reshape(-1, 1)
        stage = torch.full((len(src), 1), s / (NS - 1), device=DEV)
        L = model(src, dst, toh, dist, stage).view(k, k)
        L.fill_diagonal_(-50.0)
        return L
    Ls = [stage_logits(s) for s in range(NS)]
    total_real = sum(real_edges)
    lo, hi = -20.0, 20.0
    for _ in range(60):
        c = (lo + hi) / 2
        tot = sum(torch.sigmoid(L + c).sum().item() for L in Ls)
        if tot > total_real: hi = c
        else: lo = c
    gen_edges = [int(torch.sigmoid(L + c).sum().item()) for L in Ls]

    print("=== 발달 성장곡선: 단계별 화학시냅스 엣지 ===")
    print(f"{'단계':>4}{'실제':>8}{'생성':>8}")
    for s in range(NS):
        print(f"{s+1:>4}{real_edges[s]:>8}{gen_edges[s]:>8}")
    rg = np.corrcoef(real_edges, gen_edges)[0, 1]
    print(f"\n실제 vs 생성 성장곡선 상관: {rg:.3f}")
    print(f"실제 성장배율 {real_edges[-1]/real_edges[0]:.1f}배, 생성 {gen_edges[-1]/gen_edges[0]:.1f}배")

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, NS + 1), real_edges, "o-", label="real", lw=2)
    plt.plot(range(1, NS + 1), gen_edges, "s--", label="generated", lw=2, color="tab:red")
    plt.xlabel("developmental stage (birth→adult)"); plt.ylabel("chemical synapse edges")
    plt.title("Developmental growth: real vs generated"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{VIZ}/8_developmental.png", dpi=110); plt.close()
    print(f"시각화: {VIZ}/8_developmental.png")


if __name__ == "__main__":
    main()
