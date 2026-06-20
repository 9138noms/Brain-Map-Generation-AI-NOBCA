"""
NOMS-LAB connectome-gen : 단계 XV — 본능 회로 보존

phase8: 통계 규칙만으론 터치-회피 반사회로(앞터치→후진명령) 소실(실제0.60 vs 생성0.52).
질문: **노드 임베딩(개별 뉴런 정체성)** 이 본능 회로를 보존하나?
  규칙전용(타입+거리) vs 임베딩(+ u·v + s·s) vs 실제, escape 경로비율 비교.
  + 모델이 실제 회로 엣지(앞터치→후진)에 높은 확률을 주나 (회로를 '아는가').
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DEVN = os.path.join(HERE, "..", "data", "processed", "celegans_dev.npz")
GEN = r"D:\NOMS-LAB-D\connectome-gen\outputs\phase1_v2_generated_stage8.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TIDX = {"sensory": 0, "inter": 1, "motor": 2, "modulatory": 3, "glia": 4, "other": 5}
EMB = 16
torch.manual_seed(0); np.random.seed(0)

ANT = ["ALML", "ALMR", "AVM"]
CMD_BACK = ["AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"]
CMD_FWD = ["AVBL", "AVBR", "PVCL", "PVCR"]


class Gen(nn.Module):
    def __init__(self, N, T, use_emb):
        super().__init__()
        self.use_emb = use_emb
        self.mlp = nn.Sequential(nn.Linear(2 * T + 1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        if use_emb:
            self.u = nn.Embedding(N, EMB); self.v = nn.Embedding(N, EMB); self.s = nn.Embedding(N, EMB)
            for e in (self.u, self.v, self.s): nn.init.normal_(e.weight, std=0.1)

    def forward(self, src, dst, toh, pos, dscale):
        dist = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True) / dscale
        out = self.mlp(torch.cat([toh[src], toh[dst], dist], -1)).squeeze(-1)
        if self.use_emb:
            out = out + (self.u(src) * self.v(dst)).sum(-1) + (self.s(src) * self.s(dst)).sum(-1)
        return out


def train(model, Et, toh, P, dscale, N):
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4); lf = nn.BCEWithLogitsLoss()
    for ep in range(200):
        ps, pd = Et[:, 0], Et[:, 1]
        ns = torch.randint(0, N, (len(ps) * 5,), device=DEV); nd = torch.randint(0, N, (len(ps) * 5,), device=DEV)
        src = torch.cat([ps, ns]); dst = torch.cat([pd, nd])
        y = torch.cat([torch.ones(len(ps), device=DEV), torch.zeros(len(ns), device=DEV)])
        opt.zero_grad(); lf(model(src, dst, toh, P, dscale), y).backward(); opt.step()


def full_prob(model, N, toh, P, dscale, target, eye):
    with torch.no_grad():
        L = torch.empty(N, N, device=DEV); allj = torch.arange(N, device=DEV)
        for s0 in range(0, N, 400):
            idx = torch.arange(s0, min(s0 + 400, N), device=DEV)
            L[idx] = model(idx.repeat_interleave(N), allj.repeat(len(idx)), toh, P, dscale).view(len(idx), N)
    L[eye] = -50.0; lo, hi = -20.0, 20.0
    for _ in range(60):
        c = (lo + hi) / 2
        if torch.sigmoid(L + c).sum().item() > target: hi = c
        else: lo = c
    p = torch.sigmoid(L + c); p[eye] = 0
    return p


def main():
    d = np.load(DEVN, allow_pickle=True)
    names_all = [str(x) for x in d["node_names"]]
    g = np.load(GEN, allow_pickle=True); idx = g["node_idx"]
    names = [names_all[i] for i in idx]; loc = {n: i for i, n in enumerate(names)}
    chem = d["chem"][7]; pos7 = d["pos"][7]; types = d["node_type"]
    A_real = (chem[np.ix_(idx, idx)] > 0).astype(np.float32); np.fill_diagonal(A_real, 0)
    N = len(idx); eye = torch.eye(N, dtype=torch.bool, device=DEV)
    tix = np.array([TIDX.get(str(types[i]), 5) for i in idx]); T = 6
    toh = torch.eye(T, device=DEV)[torch.tensor(tix, device=DEV)]
    P = torch.tensor(np.nan_to_num(pos7[idx]), dtype=torch.float32, device=DEV)
    Ar = torch.tensor(A_real, device=DEV)
    ei, ej = np.where(A_real > 0); Et = torch.tensor(np.stack([ei, ej], 1), dtype=torch.long, device=DEV)
    dscale = (P[Et[:, 0]] - P[Et[:, 1]]).norm(dim=-1).mean().clamp(min=1.0)

    def mask(grp):
        m = torch.zeros(N, device=DEV)
        for n in grp:
            if n in loc: m[loc[n]] = 1.0
        return m
    ant, back, fwd = mask(ANT), mask(CMD_BACK), mask(CMD_FWD)

    def pathway(A):
        reach = A + A @ A
        b = (ant @ reach @ back).item(); f = (ant @ reach @ fwd).item()
        return b / (b + f + 1e-9)

    target = Ar.sum().item()
    pr_real = pathway(Ar)
    # 두 모델 학습 + 생성
    res = {}
    for label, use_emb in [("규칙전용", False), ("임베딩", True)]:
        m = Gen(N, T, use_emb).to(DEV); train(m, Et, toh, P, dscale, N)
        p = full_prob(m, N, toh, P, dscale, target, eye)
        prs = [pathway((torch.rand(N, N, device=DEV) < p).float() * (~eye)) for _ in range(8)]
        # 실제 회로엣지 vs forward엣지 vs 무작위에 준 확률
        ab = (ant[:, None] * back[None, :]).bool() & (Ar > 0)        # 실제 존재하는 앞→후진 엣지
        af = (ant[:, None] * fwd[None, :]).bool() & (Ar > 0)         # 실제 앞→전진 엣지
        res[label] = (np.mean(prs), p[ab].mean().item() if ab.sum() > 0 else 0,
                      p[af].mean().item() if af.sum() > 0 else 0, p[~eye].mean().item())
    rnd_pr = np.mean([pathway((torch.rand(N, N, device=DEV) < (target / (N * (N - 1)))).float() * (~eye)) for _ in range(8)])

    print(f"앞터치{[n for n in ANT if n in loc]} 후진{[n for n in CMD_BACK if n in loc]} 전진{[n for n in CMD_FWD if n in loc]}")
    print(f"\n=== escape 경로비율 [후진/(후진+전진)] (>0.5=회피회로) ===")
    print(f"  실제      : {pr_real:.3f}")
    print(f"  규칙전용  : {res['규칙전용'][0]:.3f}")
    print(f"  임베딩    : {res['임베딩'][0]:.3f}")
    print(f"  무작위    : {rnd_pr:.3f}")
    print(f"\n=== 모델이 실제 회로엣지에 준 확률 (회로를 '아는가') ===")
    print(f"{'모델':<10}{'앞→후진(회로)':>14}{'앞→전진':>12}{'평균엣지':>10}")
    for k in ["규칙전용", "임베딩"]:
        print(f"{k:<10}{res[k][1]:>14.3f}{res[k][2]:>12.3f}{res[k][3]:>10.3f}")
    e = res["임베딩"]
    print(f"\n→ 임베딩이 회로엣지에 평균보다 {e[1]/max(e[3],1e-9):.1f}배 높은 확률 주면 = 본능회로 포착")


if __name__ == "__main__":
    main()
