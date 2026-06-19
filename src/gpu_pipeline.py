"""
NOMS-LAB connectome-gen : 대규모 GPU 학습 파이프라인 (negative sampling)

목적: 전수계산 불가한 큰 connectome(초파리 14만뉴런=190억쌍) 대비.
  핵심 = N^2 쌍을 절대 만들지 않음. 양성엣지 + 무작위 음성쌍 샘플만 미니배치 학습.
  같은 코드로 벌레(213) ~ 초파리(140k) 스케일.

모델 (v2와 동일 계열, 배치화):
  logit(i→j) = MLP([type_i, type_j, dist]) + u_i·v_j(허브) + s_i·s_j(상호성)

사용:
  py -3.12 src/gpu_pipeline.py [input.npz]
  기본 입력 = data/processed/pipeline_celegans_adult.npz (make_pipeline_input.py 생성)
  초파리: 같은 포맷 npz 만들어 경로만 넘기면 됨.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(HERE, "..", "data", "processed", "pipeline_celegans_adult.npz")
MODELS = os.path.join(HERE, "..", "models")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_TYPES = 6

# --- 하이퍼파라미터 (스케일에 따라 조정) ---
EMB = 16
NEG_RATIO = 5         # 양성당 음성 샘플 수
BATCH = 4096          # 양성엣지 배치
EPOCHS = 40
LR = 5e-3
torch.manual_seed(0); np.random.seed(0)


class ScalableEdgeGen(nn.Module):
    def __init__(self, N, n_types):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * n_types + 1, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.u = nn.Embedding(N, EMB); self.v = nn.Embedding(N, EMB); self.s = nn.Embedding(N, EMB)
        for e in (self.u, self.v, self.s):
            nn.init.normal_(e.weight, std=0.1)

    def forward(self, src, dst, type_oh, pos, dscale):
        """src,dst: (B,) long. 배치 단위 logit (N^2 안 만듦)."""
        ti, tj = type_oh[src], type_oh[dst]                       # (B,6)
        dist = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True) / dscale
        feat = torch.cat([ti, tj, dist], -1)
        base = self.mlp(feat).squeeze(-1)
        emb = (self.u(src) * self.v(dst)).sum(-1) + (self.s(src) * self.s(dst)).sum(-1)
        return base + emb


def auc_ap(y, p):
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    npos = y.sum(); nneg = len(y) - npos
    auc = float("nan") if npos == 0 or nneg == 0 else (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    o = np.argsort(-p); yo = y[o]; tp = np.cumsum(yo); fp = np.cumsum(1 - yo)
    prec = tp / (tp + fp); rec = tp / max(npos, 1)
    ap = float(((rec - np.concatenate([[0], rec[:-1]])) * prec).sum())
    return auc, ap


def sample_neg(n, N, pos_set, device):
    """무작위 음성쌍 (희소그래프라 충돌 거의 없음, 표준 근사)."""
    src = torch.randint(0, N, (n,), device=device)
    dst = torch.randint(0, N, (n,), device=device)
    bad = src == dst
    dst[bad] = (dst[bad] + 1) % N
    return src, dst


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    d = np.load(path, allow_pickle=True)
    N = int(d["num_nodes"]); edges = d["edges"]; ntype = d["node_type"]; pos = d["pos"]
    n_types = int(ntype.max()) + 1
    print(f"입력: {os.path.basename(path)} | 노드 {N} | 양성엣지 {len(edges):,} | 타입 {n_types}종 | 장치 {DEV}")

    # GPU 텐서
    type_oh = torch.eye(n_types, device=DEV)[torch.tensor(ntype, device=DEV)]
    pos_t = torch.tensor(pos, dtype=torch.float32, device=DEV)
    E = torch.tensor(edges, dtype=torch.long, device=DEV)
    # train/test 분할 (양성엣지 90/10)
    perm = torch.randperm(len(E), device=DEV)
    n_te = max(1, len(E) // 10)
    te_idx, tr_idx = perm[:n_te], perm[n_te:]
    Etr, Ete = E[tr_idx], E[te_idx]
    pos_set = set(map(tuple, edges.tolist()))
    dscale = (pos_t[E[:, 0]] - pos_t[E[:, 1]]).norm(dim=-1).mean().clamp(min=1.0)

    model = ScalableEdgeGen(N, n_types).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    steps = max(1, len(Etr) // BATCH)

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); order = torch.randperm(len(Etr), device=DEV)
        for st in range(steps):
            b = order[st * BATCH:(st + 1) * BATCH]
            ps, pd = Etr[b, 0], Etr[b, 1]
            nn_ = len(b) * NEG_RATIO
            ns, nd = sample_neg(nn_, N, pos_set, DEV)
            src = torch.cat([ps, ns]); dst = torch.cat([pd, nd])
            y = torch.cat([torch.ones(len(ps), device=DEV), torch.zeros(len(ns), device=DEV)])
            opt.zero_grad()
            logit = model(src, dst, type_oh, pos_t, dscale)
            loss = lossf(logit, y); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            a, ap = evaluate(model, Ete, N, type_oh, pos_t, dscale)
            print(f"  ep{ep+1:>3}  test AUC {a:.3f}  AP {ap:.3f}  ({time.time()-t0:.1f}s)")

    a, ap = evaluate(model, Ete, N, type_oh, pos_t, dscale)
    print(f"\n[GPU 파이프라인]  test AUC {a:.3f}  AP {ap:.3f}  | 총 {time.time()-t0:.1f}s")
    os.makedirs(MODELS, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS, "gpu_pipeline.pt"))
    print(f"저장: models/gpu_pipeline.pt")
    print(f"\n초파리 스케일링: 같은 포맷 npz(num_nodes/node_type/pos/edges) 만들어")
    print(f"  py -3.12 src/gpu_pipeline.py <path> 로 실행. N^2 안 만드므로 14만뉴런도 OK.")


@torch.no_grad()
def evaluate(model, Ete, N, type_oh, pos_t, dscale):
    model.eval()
    ps, pd = Ete[:, 0], Ete[:, 1]
    ns, nd = sample_neg(len(ps) * NEG_RATIO, N, None, Ete.device)
    src = torch.cat([ps, ns]); dst = torch.cat([pd, nd])
    y = torch.cat([torch.ones(len(ps)), torch.zeros(len(ns))]).numpy()
    p = torch.sigmoid(model(src, dst, type_oh, pos_t, dscale)).cpu().numpy()
    return auc_ap(y, p)


if __name__ == "__main__":
    main()
