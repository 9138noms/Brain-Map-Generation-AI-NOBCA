"""
NOMS-LAB connectome-gen : Phase 1 v2 — 노드 임베딩 + 상호성 (GPU 전용/벡터화)

v1 한계(상호성 0.28→0.05, 허브 부족) 해결 시도:
  logit(i→j) = MLP(type_i, type_j, dist, stage)      # 일반 배선규칙
             + u_i · v_j                              # 방향 노드임베딩 → 허브성
             + s_i · s_j                              # 대칭 노드임베딩 → 상호성

설계 원칙:
  - 모든 연산 GPU 텐서 (CPU 이중루프 제거, 초파리 스케일 대비 벡터화)
  - 단계 1~7 학습 → 성체(8) 예측 AUC/AP + 생성 구조비교 (v1과 동일 점수판)
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
NPZ = os.path.join(HERE, "..", "data", "processed", "celegans_dev.npz")
MODELS = os.path.join(HERE, "..", "models")
OUTPUTS = r"D:\NOMS-LAB-D\connectome-gen\outputs"

TYPES = ["sensory", "inter", "motor", "modulatory", "glia", "other"]
TIDX = {t: i for i, t in enumerate(TYPES)}
N_STAGES = 8
EMB = 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def auc_score(y, score):
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return float("nan") if npos == 0 or nneg == 0 else (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def ap_score(y, score):
    order = np.argsort(-score); yo = y[order]
    tp = np.cumsum(yo); fp = np.cumsum(1 - yo)
    prec = tp / (tp + fp); rec = tp / max(y.sum(), 1)
    return float(((rec - np.concatenate([[0], rec[:-1]])) * prec).sum())


def graph_stats(A, tix):
    k = A.shape[0]; n = A.sum()
    out = A.sum(1); inn = A.sum(0)
    si, ii, mi = TIDX["sensory"], TIDX["inter"], TIDX["motor"]
    src = tix[:, None]; dst = tix[None, :]
    return dict(edges=int(n), density=float(n / (k * (k - 1))),
                reciprocity=float((A * A.T).sum() / max(n, 1)),
                mean_deg=float(out.mean()), max_outdeg=int(out.max()), max_indeg=int(inn.max()),
                sensory2inter=float(A[(src == si) & (dst == ii)].sum() / max(n, 1)),
                inter2motor=float(A[(src == ii) & (dst == mi)].sum() / max(n, 1)))


class EdgeGen(nn.Module):
    """노드 임베딩 + 상호성. forward는 한 단계의 전체 (k,k) logit 행렬."""
    def __init__(self, N):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(14, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.u = nn.Embedding(N, EMB)   # source (허브-out)
        self.v = nn.Embedding(N, EMB)   # target (허브-in)
        self.s = nn.Embedding(N, EMB)   # symmetric (상호성)
        for e in (self.u, self.v, self.s):
            nn.init.normal_(e.weight, std=0.1)

    def forward(self, idx, dist, snorm):
        k = len(idx)
        ti = self.type_oh[idx]                                   # (k,6)
        feat = torch.cat([
            ti[:, None, :].expand(k, k, -1),
            ti[None, :, :].expand(k, k, -1),
            dist[:, :, None],
            torch.full((k, k, 1), snorm, device=idx.device),
        ], -1)                                                   # (k,k,14)
        base = self.mlp(feat.reshape(-1, 14)).reshape(k, k)
        u, v, s = self.u(idx), self.v(idx), self.s(idx)
        logit = base + u @ v.t() + s @ s.t()
        return logit


def main():
    d = np.load(NPZ, allow_pickle=True)
    chem = d["chem"]; pos = d["pos"]; present = d["present"]; types = d["node_type"]
    N = len(types)
    tix = np.array([TIDX.get(str(t), TIDX["other"]) for t in types])

    # 단계별 (존재&좌표) 노드 + 거리행렬 GPU 텐서로 사전계산
    stage_idx, stage_dist = [], []
    all_d = []
    for s in range(N_STAGES):
        ok = present[s] & ~np.isnan(pos[s, :, 0]); idx = np.where(ok)[0]
        P = torch.tensor(pos[s, idx], dtype=torch.float32, device=DEV)
        dist = torch.cdist(P, P)
        stage_idx.append(torch.tensor(idx, device=DEV))
        stage_dist.append(dist)
        all_d.append(dist[dist > 0])
    dscale = torch.cat(all_d).mean()                 # 거리 정규화 스케일
    stage_dist = [dm / dscale for dm in stage_dist]

    model = EdgeGen(N).to(DEV)
    model.type_oh = torch.eye(len(TYPES), device=DEV)[torch.tensor(tix, device=DEV)]
    chem_t = torch.tensor(chem, device=DEV)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

    # pos_weight (전체 학습단계 평균 불균형)
    npos = sum((chem_t[s][torch.meshgrid(stage_idx[s], stage_idx[s], indexing="ij")] > 0).sum().item() for s in range(7))
    ntot = sum(len(stage_idx[s]) ** 2 for s in range(7))
    posw = torch.tensor((ntot - npos) / max(npos, 1), device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=posw)

    def stage_target(s):
        ii, jj = torch.meshgrid(stage_idx[s], stage_idx[s], indexing="ij")
        return (chem_t[s][ii, jj] > 0).float()

    for ep in range(300):
        model.train(); opt.zero_grad(); loss = 0.0
        for s in range(7):                                       # 단계 1~7
            k = len(stage_idx[s])
            logit = model(stage_idx[s], stage_dist[s], s / (N_STAGES - 1))
            mask = ~torch.eye(k, dtype=torch.bool, device=DEV)   # 자기연결 제외
            loss = loss + lossf(logit[mask], stage_target(s)[mask])
        loss.backward(); opt.step()
        if (ep + 1) % 60 == 0:
            model.eval()
            with torch.no_grad():
                k = len(stage_idx[7])
                lg = model(stage_idx[7], stage_dist[7], 7 / (N_STAGES - 1))
                mask = ~torch.eye(k, dtype=torch.bool, device=DEV)
                p = torch.sigmoid(lg[mask]).cpu().numpy(); y = stage_target(7)[mask].cpu().numpy()
            print(f"  ep{ep+1:>3}  test AUC {auc_score(y, p):.3f}  AP {ap_score(y, p):.3f}")

    # ---- 최종 평가 (성체=단계8) ----
    model.eval()
    with torch.no_grad():
        k = len(stage_idx[7])
        lg = model(stage_idx[7], stage_dist[7], 7 / (N_STAGES - 1)).cpu().numpy()
    idx8 = stage_idx[7].cpu().numpy()
    A_real = (chem[7][np.ix_(idx8, idx8)] > 0).astype(np.float32); np.fill_diagonal(A_real, 0)
    eye = np.eye(k, dtype=bool)
    y = A_real[~eye]; p_eval = (1 / (1 + np.exp(-lg)))[~eye]
    print(f"\n[v2 노드임베딩+상호성]  AUC {auc_score(y, p_eval):.3f}  AP {ap_score(y, p_eval):.3f}")

    # ---- 생성 (밀도 맞춤) ----
    zg = lg.copy(); np.fill_diagonal(zg, -50.0)
    target = float(A_real.sum()); lo, hi = -20.0, 20.0
    for _ in range(60):
        c = (lo + hi) / 2
        hi, lo = (c, lo) if (1 / (1 + np.exp(-(zg + c)))).sum() > target else (hi, c)
    pg = 1 / (1 + np.exp(-(zg + c))); np.fill_diagonal(pg, 0)
    A_gen = (np.random.rand(k, k) < pg).astype(np.float32); np.fill_diagonal(A_gen, 0)

    tix8 = tix[idx8]; sr = graph_stats(A_real, tix8); sg = graph_stats(A_gen, tix8)
    print(f"\n=== 생성 vs 실제 (단계8 성체, {k}뉴런) ===")
    print(f"{'지표':<14}{'실제':>10}{'생성v2':>10}")
    for key in ["edges", "density", "reciprocity", "mean_deg", "max_outdeg", "max_indeg", "sensory2inter", "inter2motor"]:
        rv, gv = sr[key], sg[key]
        f = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
        print(f"{key:<14}{f(rv):>10}{f(gv):>10}")

    os.makedirs(MODELS, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS, "phase1_v2.pt"))
    os.makedirs(OUTPUTS, exist_ok=True)
    np.savez_compressed(os.path.join(OUTPUTS, "phase1_v2_generated_stage8.npz"),
                        A_real=A_real, A_gen=A_gen, prob=pg, node_idx=idx8, types_idx=tix8)
    print(f"\n저장: models/phase1_v2.pt, D:\\...\\outputs\\phase1_v2_generated_stage8.npz")


if __name__ == "__main__":
    main()
