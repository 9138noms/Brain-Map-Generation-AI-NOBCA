"""
NOMS-LAB connectome-gen : 단계 XVI (Tier-A #1) — 가중치 생성 (시냅스 수)

지금까지 0/1 이진 생성 → 이제 **시냅스 수(세기)** 까지 생성.
  - 존재 모델: P(edge) (기존, 임베딩)
  - 가중치 모델: 존재하는 엣지의 log(시냅스수) 회귀 (타입+거리+임베딩)
  - 생성: 존재 샘플 → 각 엣지에 시냅스수 부여
검증: 가중치 분포(평균/최대/강한엣지비율)가 실제와 같나.
이게 본능 방향비대칭(#2)의 토대.
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


class WeightedGen(nn.Module):
    """존재 logit + log(시냅스수) 동시 출력."""
    def __init__(self, N, T):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(2 * T + 1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.exist = nn.Linear(64, 1)
        self.weight = nn.Linear(64 + EMB, 1)
        self.u = nn.Embedding(N, EMB); self.v = nn.Embedding(N, EMB); self.s = nn.Embedding(N, EMB)
        self.wu = nn.Embedding(N, EMB)
        for e in (self.u, self.v, self.s, self.wu): nn.init.normal_(e.weight, std=0.1)

    def forward(self, src, dst, toh, pos, dscale):
        dist = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True) / dscale
        h = self.trunk(torch.cat([toh[src], toh[dst], dist], -1))
        ex = self.exist(h).squeeze(-1) + (self.u(src) * self.v(dst)).sum(-1) + (self.s(src) * self.s(dst)).sum(-1)
        wt = self.weight(torch.cat([h, self.wu(src) * self.wu(dst)], -1)).squeeze(-1)  # log-count
        return ex, wt


def main():
    d = np.load(DEVN, allow_pickle=True)
    g = np.load(GEN, allow_pickle=True); idx = g["node_idx"]
    chem = d["chem"][7]; pos7 = d["pos"][7]; types = d["node_type"]
    W_real = chem[np.ix_(idx, idx)].astype(np.float32); np.fill_diagonal(W_real, 0)
    A_real = (W_real > 0).astype(np.float32)
    N = len(idx); T = 6; eye = torch.eye(N, dtype=torch.bool, device=DEV)
    tix = np.array([TIDX.get(str(types[i]), 5) for i in idx])
    toh = torch.eye(T, device=DEV)[torch.tensor(tix, device=DEV)]
    P = torch.tensor(np.nan_to_num(pos7[idx]), dtype=torch.float32, device=DEV)
    Wt = torch.tensor(W_real, device=DEV); At = torch.tensor(A_real, device=DEV)
    ei, ej = np.where(A_real > 0)
    Et = torch.tensor(np.stack([ei, ej], 1), dtype=torch.long, device=DEV)
    wlog = torch.log(torch.tensor(W_real[ei, ej], device=DEV))   # 실제 log 시냅스수
    dscale = (P[Et[:, 0]] - P[Et[:, 1]]).norm(dim=-1).mean().clamp(min=1.0)

    model = WeightedGen(N, T).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(); mse = nn.MSELoss()
    for ep in range(300):
        ps, pd = Et[:, 0], Et[:, 1]
        ns = torch.randint(0, N, (len(ps) * 5,), device=DEV); nd = torch.randint(0, N, (len(ps) * 5,), device=DEV)
        src = torch.cat([ps, ns]); dst = torch.cat([pd, nd])
        y = torch.cat([torch.ones(len(ps), device=DEV), torch.zeros(len(ns), device=DEV)])
        ex, wt = model(src, dst, toh, P, dscale)
        loss = bce(ex, y) + mse(wt[:len(ps)], wlog)            # 가중치는 존재엣지만
        opt.zero_grad(); loss.backward(); opt.step()

    # 전수 → 밀도맞춤 존재 → 가중치 부여
    with torch.no_grad():
        Lex = torch.empty(N, N, device=DEV); Lwt = torch.empty(N, N, device=DEV); allj = torch.arange(N, device=DEV)
        for s0 in range(0, N, 400):
            i = torch.arange(s0, min(s0 + 400, N), device=DEV)
            ex, wt = model(i.repeat_interleave(N), allj.repeat(len(i)), toh, P, dscale)
            Lex[i] = ex.view(len(i), N); Lwt[i] = wt.view(len(i), N)
    Lex[eye] = -50.0; target = At.sum().item(); lo, hi = -20.0, 20.0
    for _ in range(60):
        c = (lo + hi) / 2
        if torch.sigmoid(Lex + c).sum().item() > target: hi = c
        else: lo = c
    A_gen = (torch.rand(N, N, device=DEV) < torch.sigmoid(Lex + c)).float(); A_gen[eye] = 0
    W_gen = A_gen * torch.exp(Lwt).clamp(1, 200)               # 생성 시냅스수

    def wstats(W, A):
        w = W[A > 0]
        return float(w.mean()), float(w.max()), float(w.std()), float((w >= 5).float().mean())
    rm, rx, rs, rf = wstats(Wt, At); gm, gx, gs, gf = wstats(W_gen, A_gen)
    # 타입쌍 가중치 행렬 상관
    def tw(W, A):
        M = torch.zeros(T, T, device=DEV); cnt = torch.zeros(T, T, device=DEV)
        s, dd = (A > 0).nonzero(as_tuple=True)
        ti = torch.tensor(tix, device=DEV)
        M.index_put_((ti[s], ti[dd]), W[s, dd], accumulate=True)
        cnt.index_put_((ti[s], ti[dd]), torch.ones(len(s), device=DEV), accumulate=True)
        return (M / cnt.clamp(min=1)).flatten()
    a = tw(Wt, At) - tw(Wt, At).mean(); b = tw(W_gen, A_gen) - tw(W_gen, A_gen).mean()
    tw_corr = float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-8))

    print("=== 가중치(시냅스 수) 생성 검증 (벌레) ===")
    print(f"{'':<14}{'평균':>8}{'최대':>8}{'표준편차':>9}{'강한엣지(≥5)':>13}")
    print(f"{'실제':<14}{rm:>8.2f}{rx:>8.0f}{rs:>9.2f}{rf*100:>11.1f}%")
    print(f"{'생성':<14}{gm:>8.2f}{gx:>8.0f}{gs:>9.2f}{gf*100:>11.1f}%")
    print(f"\n타입쌍 평균가중치 실제와 상관: {tw_corr:.3f}")
    print(f"→ 이제 0/1 아니라 *세기*까지 생성. 본능 방향비대칭(#2) 토대 마련.")
    np.savez_compressed(r"D:\NOMS-LAB-D\connectome-gen\outputs\phase16_weighted.npz",
                        W_gen=W_gen.cpu().numpy())


if __name__ == "__main__":
    main()
