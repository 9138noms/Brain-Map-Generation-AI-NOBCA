"""
NOMS-LAB connectome-gen : Phase 1 — 엣지 단위 생성 모델 (v1)

아이디어: 그래프 한 장=샘플1개(8개뿐, 과적합) 대신,
          연결 하나=샘플1개 로 학습. 수십만 엣지결정이 학습신호가 됨.
  P(i→j 화학시냅스 존재 | type_i, type_j, 3D거리, 발달단계)

평가:
  1) 예측: 단계 1~7 학습 → 단계 8(성체) 엣지 예측 AUC/AP (진짜 규칙 배웠나)
  2) 생성: 단계 8 connectome을 샘플링으로 생성 → 실제와 구조통계 비교
  baseline: 거리(가까울수록 연결) 만으로 예측

출력:
  C .../models/phase1_edge_mlp.pt
  D .../outputs/phase1_generated_stage8.npz  (+ 콘솔 리포트)
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
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)


# ---------------- 데이터셋 구성 (엣지=샘플) ----------------
def build_pairs(d):
    """모든 단계의 (존재+좌표보유) 뉴런 쌍 i!=j 에 대해 특징/라벨 생성."""
    chem = d["chem"]; pos = d["pos"]; present = d["present"]
    types = d["node_type"]; N = len(types)
    type_oh = np.zeros((N, len(TYPES)), np.float32)
    for i, t in enumerate(types):
        type_oh[i, TIDX.get(str(t), TIDX["other"])] = 1.0

    feats, labels, stages, pair_ix = [], [], [], []
    for s in range(N_STAGES):
        ok = present[s] & ~np.isnan(pos[s, :, 0])  # 존재 & 좌표有
        idx = np.where(ok)[0]
        P = pos[s, idx]                              # (k,3)
        # 모든 순서쌍 거리
        diff = P[:, None, :] - P[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))         # (k,k)
        snorm = s / (N_STAGES - 1)
        for a in range(len(idx)):
            for b in range(len(idx)):
                if a == b:
                    continue
                i, j = idx[a], idx[b]
                feats.append(np.concatenate([type_oh[i], type_oh[j],
                                             [dist[a, b], snorm]]))
                labels.append(1.0 if chem[s, i, j] > 0 else 0.0)
                stages.append(s)
                pair_ix.append((s, i, j))
    X = np.array(feats, np.float32)
    y = np.array(labels, np.float32)
    st = np.array(stages, np.int64)
    return X, y, st, pair_ix


# ---------------- 평가 지표 ----------------
def auc_score(y, score):
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def ap_score(y, score):
    order = np.argsort(-score)
    yo = y[order]
    tp = np.cumsum(yo); fp = np.cumsum(1 - yo)
    prec = tp / (tp + fp); rec = tp / max(y.sum(), 1)
    rec_prev = np.concatenate([[0], rec[:-1]])
    return float(((rec - rec_prev) * prec).sum())


# ---------------- 모델 ----------------
class EdgeMLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def standardize(X, mu, sd):
    return (X - mu) / sd


# ---------------- 구조통계 (생성 평가용) ----------------
def graph_stats(A, types_idx):
    """A: (k,k) 0/1 인접행렬. 밀도/상호성/평균차수/타입흐름."""
    k = A.shape[0]
    n_edge = A.sum()
    density = n_edge / (k * (k - 1))
    recip = (A * A.T).sum() / max(n_edge, 1)            # i→j 이면서 j→i
    outdeg = A.sum(1); indeg = A.sum(0)
    # 타입 흐름: sensory→inter, inter→motor 비율
    si, ii, mi = TIDX["sensory"], TIDX["inter"], TIDX["motor"]
    src = types_idx[:, None]; dst = types_idx[None, :]
    s2i = A[(src == si) & (dst == ii)].sum() / max(n_edge, 1)
    i2m = A[(src == ii) & (dst == mi)].sum() / max(n_edge, 1)
    return dict(edges=int(n_edge), density=float(density), reciprocity=float(recip),
                mean_deg=float(outdeg.mean()), max_outdeg=int(outdeg.max()),
                max_indeg=int(indeg.max()), sensory2inter=float(s2i),
                inter2motor=float(i2m))


def main():
    d = np.load(NPZ, allow_pickle=True)
    X, y, st, pair_ix = build_pairs(d)
    print(f"엣지 샘플: {len(y):,}  (양성 {int(y.sum()):,} = {y.mean()*100:.1f}%)")

    tr = st <= 6          # 단계 1~7 학습 (0-index: 0~6)
    te = st == 7          # 단계 8 평가
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr = standardize(X[tr], mu, sd); Xte = standardize(X[te], mu, sd)
    ytr, yte = y[tr], y[te]

    # --- baseline: 거리만 (가까울수록 연결) ---
    dist_col = -X[te][:, -2]          # 거리 특징(끝에서 두번째), 부호반전
    print(f"\n[baseline 거리만]  AUC {auc_score(yte, dist_col):.3f}  "
          f"AP {ap_score(yte, dist_col):.3f}")

    # --- MLP 학습 ---
    Xt = torch.tensor(Xtr, device=DEV); yt = torch.tensor(ytr, device=DEV)
    Xv = torch.tensor(Xte, device=DEV)
    model = EdgeMLP(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    posw = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=posw)
    n = len(Xt); bs = 8192
    for ep in range(40):
        model.train(); perm = torch.randperm(n, device=DEV)
        for k in range(0, n, bs):
            b = perm[k:k + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b]); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xv)).cpu().numpy()
            print(f"  ep{ep+1:>2}  test AUC {auc_score(yte, p):.3f}  "
                  f"AP {ap_score(yte, p):.3f}")

    model.eval()
    with torch.no_grad():
        ptest = torch.sigmoid(model(Xv)).cpu().numpy()
    print(f"\n[MLP type+거리+단계]  AUC {auc_score(yte, ptest):.3f}  "
          f"AP {ap_score(yte, ptest):.3f}")

    os.makedirs(MODELS, exist_ok=True)
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd},
               os.path.join(MODELS, "phase1_edge_mlp.pt"))

    # ---------------- 생성: 단계8 connectome 샘플링 ----------------
    types = d["node_type"]; types_idx = np.array([TIDX.get(str(t), TIDX["other"]) for t in types])
    pos = d["pos"]; present = d["present"]; chem = d["chem"]
    s = 7
    ok = present[s] & ~np.isnan(pos[s, :, 0]); idx = np.where(ok)[0]
    k = len(idx)
    # 실제 단계8 인접행렬 (0/1)
    A_real = (chem[s][np.ix_(idx, idx)] > 0).astype(np.float32)
    np.fill_diagonal(A_real, 0)

    # 생성: 각 쌍의 예측확률로 Bernoulli 샘플
    P = pos[s, idx]; diff = P[:, None, :] - P[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    type_oh = np.eye(len(TYPES), dtype=np.float32)[types_idx[idx]]
    feats = []
    for a in range(k):
        for b in range(k):
            feats.append(np.concatenate([type_oh[a], type_oh[b], [dist[a, b], 1.0]]))
    Fg = standardize(np.array(feats, np.float32), mu, sd)
    with torch.no_grad():
        zg = model(torch.tensor(Fg, device=DEV)).cpu().numpy().reshape(k, k)
    np.fill_diagonal(zg, -50.0)  # 자기연결 차단
    # pos_weight 때문에 확률이 뻥튀기됨 → 기대 엣지수가 실제와 같아지도록
    # logit 바이어스 c 를 이분탐색해 밀도 맞춤 생성 (공정 비교)
    target = float(A_real.sum())
    lo, hi = -20.0, 20.0
    for _ in range(60):
        c = (lo + hi) / 2
        if (1 / (1 + np.exp(-(zg + c)))).sum() > target:
            hi = c
        else:
            lo = c
    pg = 1 / (1 + np.exp(-(zg + c)))
    np.fill_diagonal(pg, 0)
    print(f"\n밀도맞춤 logit bias c={c:.2f}, 기대엣지 {pg.sum():.0f} (목표 {target:.0f})")
    A_gen = (np.random.rand(k, k) < pg).astype(np.float32)
    np.fill_diagonal(A_gen, 0)

    tix = types_idx[idx]
    sr = graph_stats(A_real, tix); sg = graph_stats(A_gen, tix)
    print(f"\n=== 생성 vs 실제 (단계8 성체, {k}뉴런) ===")
    print(f"{'지표':<14}{'실제':>10}{'생성':>10}")
    for key in ["edges", "density", "reciprocity", "mean_deg", "max_outdeg",
                "max_indeg", "sensory2inter", "inter2motor"]:
        rv, gv = sr[key], sg[key]
        fr = f"{rv:.3f}" if isinstance(rv, float) else str(rv)
        fg = f"{gv:.3f}" if isinstance(gv, float) else str(gv)
        print(f"{key:<14}{fr:>10}{fg:>10}")

    os.makedirs(OUTPUTS, exist_ok=True)
    np.savez_compressed(os.path.join(OUTPUTS, "phase1_generated_stage8.npz"),
                        A_real=A_real, A_gen=A_gen, prob=pg,
                        node_idx=idx, types_idx=tix)
    print(f"\n저장: {os.path.join(OUTPUTS, 'phase1_generated_stage8.npz')}")


if __name__ == "__main__":
    main()
