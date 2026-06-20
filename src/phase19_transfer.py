"""
NOMS-LAB connectome-gen : 단계 XIX (Tier-A #5) — 종간 전이

질문: 한 종에서 배운 공간 배선규칙이 다른 종을 예측하나?
거리(연결거리로 정규화)→P(연결) 규칙을 벌레/쥐 각각 학습 → 서로 교차 예측 AUC.
교차 ≈ 종내 면 = 배선 원리가 종 초월. (벌레·쥐만 위치정보 보유)
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
GENW = r"D:\NOMS-LAB-D\connectome-gen\outputs\phase1_v2_generated_stage8.npz"
MOUSE = os.path.join(HERE, "..", "data", "processed", "pipeline_microns_mouse.npz")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    np_ = y.sum(); nn_ = len(y) - np_
    return (r[y == 1].sum() - np_ * (np_ + 1) / 2) / (np_ * nn_)


def build_pairs(pos, edges, n_neg_per_pos=1):
    """거리(연결거리 정규화) 특징 + 라벨. 균형 샘플."""
    P = torch.tensor(pos, dtype=torch.float32, device=DEV)
    N = len(P); E = torch.tensor(edges, dtype=torch.long, device=DEV)
    edist = (P[E[:, 0]] - P[E[:, 1]]).norm(dim=-1)
    scale = edist.mean().clamp(min=1.0)
    npos = len(E)
    # 음성: 무작위 쌍
    ns = torch.randint(0, N, (npos * n_neg_per_pos,), device=DEV)
    nd = torch.randint(0, N, (npos * n_neg_per_pos,), device=DEV)
    ndist = (P[ns] - P[nd]).norm(dim=-1)
    X = torch.cat([edist, ndist]) / scale
    y = torch.cat([torch.ones(npos, device=DEV), torch.zeros(len(ns), device=DEV)])
    return X.unsqueeze(1), y


class DistRule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def train(X, y):
    m = DistRule().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=5e-3); lf = nn.BCEWithLogitsLoss()
    for _ in range(400):
        opt.zero_grad(); lf(m(X), y).backward(); opt.step()
    return m


def main():
    # 벌레
    d = np.load(DEVN, allow_pickle=True); idx = np.load(GENW, allow_pickle=True)["node_idx"]
    chem = d["chem"][7]; pos7 = np.nan_to_num(d["pos"][7])
    wpos = pos7[idx]
    sub = chem[np.ix_(idx, idx)]; np.fill_diagonal(sub, 0); wi, wj = np.where(sub > 0)
    Xw, yw = build_pairs(wpos, np.stack([wi, wj], 1))
    # 쥐
    m = np.load(MOUSE, allow_pickle=True)
    Xm, ym = build_pairs(m["pos"], m["edges"])

    mw = train(Xw, yw); mm = train(Xm, ym)
    with torch.no_grad():
        ww = auc(yw.cpu().numpy(), mw(Xw).cpu().numpy())
        wm = auc(ym.cpu().numpy(), mw(Xm).cpu().numpy())   # 벌레규칙→쥐
        mm_ = auc(ym.cpu().numpy(), mm(Xm).cpu().numpy())
        mw_ = auc(yw.cpu().numpy(), mm(Xw).cpu().numpy())   # 쥐규칙→벌레

    print("=== 종간 전이: 공간 배선규칙 AUC ===")
    print(f"{'학습\\테스트':<12}{'벌레':>8}{'쥐':>8}")
    print(f"{'벌레규칙':<12}{ww:>8.3f}{wm:>8.3f}")
    print(f"{'쥐규칙':<12}{mw_:>8.3f}{mm_:>8.3f}")
    print(f"\n교차(벌레→쥐) {wm:.3f}, (쥐→벌레) {mw_:.3f} vs 종내 벌레 {ww:.3f}/쥐 {mm_:.3f}")
    cross = (wm + mw_) / 2; within = (ww + mm_) / 2
    print(f"→ 교차 {cross:.3f} / 종내 {within:.3f}: ", end="")
    print("공간 배선원리 종 초월" if cross > 0.55 and cross > within - 0.1 else "전이 제한적")


if __name__ == "__main__":
    main()
