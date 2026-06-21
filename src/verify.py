"""
NOMS-LAB connectome-gen : 공개 전 검증 (대조군 + 다중시드)

핵심 주장이 재현되나 + 아티팩트 아닌가 확인:
  C1 구조충실도 (타입흐름) — 생성 vs 실제, 여러 샘플, +무작위 대조
  C2 새로움 — 생성-실제 겹침 vs 자연변이 vs 무작위
  C3 종간보편 — 벌레↔쥐 거리규칙 교차예측, 다중시드, +셔플 대조(→0.5여야)
각 주장에 PASS/주의 표시.
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
TIDX = {"sensory": 0, "inter": 1, "motor": 2, "modulatory": 3, "glia": 4, "other": 5}


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def jaccard(A, B):
    a = A > 0; b = B > 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def typeflow(A, t, T):
    M = np.zeros((T, T)); s, d = np.where(A > 0)
    for i, j in zip(t[s], t[d]): M[i, j] += 1
    return (M / max(M.sum(), 1)).flatten()


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = y.sum(); n = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


# ---------- C1 구조충실도 + C2 새로움 ----------
def c1_c2():
    d = np.load(DEVN, allow_pickle=True); g = np.load(GENW, allow_pickle=True)
    idx = g["node_idx"]; prob = g["prob"]; types = d["node_type"]; chem = d["chem"]
    A8 = (chem[7][np.ix_(idx, idx)] > 0).astype(float); np.fill_diagonal(A8, 0)
    A7 = (chem[6][np.ix_(idx, idx)] > 0).astype(float); np.fill_diagonal(A7, 0)
    tix = np.array([TIDX.get(str(types[i]), 5) for i in idx]); T = 6; k = len(idx)
    eye = np.eye(k, dtype=bool); tgt = int(A8.sum())
    Mr = typeflow(A8, tix, T)
    tf_g, jac_g, tf_r, jac_r = [], [], [], []
    for s in range(5):
        rng = np.random.default_rng(s)
        Ag = (rng.random((k, k)) < prob).astype(float); Ag[eye] = 0
        Arnd = np.zeros(k * k); Arnd[rng.choice(k * k, tgt, replace=False)] = 1
        Arnd = Arnd.reshape(k, k); Arnd[eye] = 0
        tf_g.append(corr(typeflow(Ag, tix, T), Mr)); jac_g.append(jaccard(Ag, A8))
        tf_r.append(corr(typeflow(Arnd, tix, T), Mr)); jac_r.append(jaccard(Arnd, A8))
    print("=== C1 구조충실도 (타입흐름 상관, 5시드 평균±std) ===")
    print(f"  생성 {np.mean(tf_g):.3f}±{np.std(tf_g):.3f}  vs  무작위 {np.mean(tf_r):.3f}±{np.std(tf_r):.3f}")
    print(f"  판정: {'PASS' if np.mean(tf_g) > np.mean(tf_r) + 0.2 else '주의'}")
    print("=== C2 새로움 (실제와 겹침 Jaccard) ===")
    print(f"  생성 {np.mean(jac_g):.3f}  | 자연변이(실제7vs8) {jaccard(A7,A8):.3f}  | 무작위 {np.mean(jac_r):.3f}")
    nv = jaccard(A7, A8)
    print(f"  판정: {'PASS (무작위<<생성<=자연변이)' if np.mean(jac_r) < np.mean(jac_g) <= nv + 0.05 else '주의'}")


# ---------- C3 종간보편 ----------
class DistRule(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def pairs(pos, edges, shuffle=False, seed=0):
    rng = np.random.default_rng(seed)
    P = torch.tensor(pos, dtype=torch.float32, device=DEV); N = len(P)
    E = torch.tensor(edges, dtype=torch.long, device=DEV)
    if shuffle:  # 셔플 대조: 엣지 무작위화 → 거리신호 파괴
        E = torch.randint(0, N, E.shape, device=DEV)
    ed = (P[E[:, 0]] - P[E[:, 1]]).norm(dim=-1); sc = ed.mean().clamp(min=1)
    ns = torch.randint(0, N, (len(E),), device=DEV); nd = torch.randint(0, N, (len(E),), device=DEV)
    nde = (P[ns] - P[nd]).norm(dim=-1)
    X = torch.cat([ed, nde]).unsqueeze(1) / sc
    y = torch.cat([torch.ones(len(E)), torch.zeros(len(E))])
    return X.to(DEV), y


def train(X, y):
    m = DistRule().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=5e-3); lf = nn.BCEWithLogitsLoss()
    yd = y.to(DEV)
    for _ in range(400): opt.zero_grad(); lf(m(X), yd).backward(); opt.step()
    return m


def c3():
    d = np.load(DEVN, allow_pickle=True); idx = np.load(GENW, allow_pickle=True)["node_idx"]
    chem = d["chem"][7]; wp = np.nan_to_num(d["pos"][7])[idx]
    sub = chem[np.ix_(idx, idx)]; np.fill_diagonal(sub, 0); wi, wj = np.where(sub > 0)
    we = np.stack([wi, wj], 1)
    m = np.load(MOUSE, allow_pickle=True); mp = m["pos"]; me = m["edges"]
    cross, within, shuf = [], [], []
    for s in range(3):
        torch.manual_seed(s)
        Xw, yw = pairs(wp, we, seed=s); Xm, ym = pairs(mp, me, seed=s)
        mw = train(Xw, yw); mm = train(Xm, ym)
        with torch.no_grad():
            within.append(auc(ym.numpy(), mm(Xm).cpu().numpy()))           # 쥐→쥐
            cross.append(auc(ym.numpy(), mw(Xm).cpu().numpy()))            # 벌레→쥐
        Xsh, ysh = pairs(mp, me, shuffle=True, seed=s)                     # 셔플 대조
        msh = train(Xsh, ysh)
        with torch.no_grad():
            shuf.append(auc(ysh.numpy(), msh(Xsh).cpu().numpy()))
    print("=== C3 종간보편 (거리규칙 AUC, 3시드) ===")
    print(f"  종내(쥐→쥐) {np.mean(within):.3f}  교차(벌레→쥐) {np.mean(cross):.3f}  셔플대조 {np.mean(shuf):.3f}")
    ok = abs(np.mean(cross) - np.mean(within)) < 0.05 and np.mean(cross) > 0.6 and np.mean(shuf) < 0.55
    print(f"  판정: {'PASS (교차≈종내, 셔플~0.5)' if ok else '주의'}")


def main():
    c1_c2(); print(); c3()


if __name__ == "__main__":
    main()
