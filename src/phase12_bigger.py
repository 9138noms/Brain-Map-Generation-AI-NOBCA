"""
NOMS-LAB connectome-gen : 단계 XII — 학습보다 큰 뇌 생성 + "어떻게 만드나" 분석

쥐 피질(2220뉴런) 학습 → 그 배선규칙으로 **22배 큰 뇌(50,000뉴런)** 생성.
AI가 뇌를 만드는 방식 = 배운 규칙(타입친화도 + 거리커널)을 더 많은 뉴런에 적용.
  - 학습한 규칙 추출/시각화 (레시피)
  - 큰 뇌 생성 (공간-국소 희소생성, 엣지 반환)
  - 큰 뇌 구조가 실제 쥐 규칙을 유지하나 (타입흐름/차수/거리의존성)
출력: D/outputs/large/ (npz + viz)
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
DOUT = r"D:\NOMS-LAB-D\connectome-gen\outputs\large"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
N_BIG = 50_000


class Rule(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.mlp = nn.Sequential(nn.Linear(2 * T + 1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, ti, tj, dist):
        return self.mlp(torch.cat([ti, tj, dist], -1)).squeeze(-1)


def main():
    d = np.load(MOUSE, allow_pickle=True)
    N = int(d["num_nodes"]); E = d["edges"]; nt = d["node_type"]; pos = d["pos"]
    T = int(nt.max()) + 1; vocab = [str(x) for x in d["type_vocab"]]
    type_dist = np.bincount(nt, minlength=T) / N
    P = torch.tensor(pos, dtype=torch.float32, device=DEV)
    toh = torch.eye(T, device=DEV)[torch.tensor(nt, device=DEV)]
    Et = torch.tensor(E, dtype=torch.long, device=DEV)
    dscale = (P[Et[:, 0]] - P[Et[:, 1]]).norm(dim=-1).mean().clamp(min=1.0)
    mean_deg = len(E) / N
    span = (P.max(0).values - P.min(0).values).clamp(min=1.0)
    dvol = N / span.prod().item()
    print(f"학습데이터: 쥐 {N}뉴런, 평균차수 {mean_deg:.1f}, 거리스케일 {dscale:.0f}nm")

    # --- 규칙 학습 ---
    model = Rule(T).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(120):
        ps, pd = Et[:, 0], Et[:, 1]
        ns = torch.randint(0, N, (len(ps) * 5,), device=DEV); nd = torch.randint(0, N, (len(ps) * 5,), device=DEV)
        src = torch.cat([ps, ns]); dst = torch.cat([pd, nd])
        dist = (P[src] - P[dst]).norm(dim=-1, keepdim=True) / dscale
        y = torch.cat([torch.ones(len(ps), device=DEV), torch.zeros(len(ns), device=DEV)])
        opt.zero_grad(); lossf(model(toh[src], toh[dst], dist), y).backward(); opt.step()

    # --- 규칙 추출 (레시피) ---
    EYE = torch.eye(T, device=DEV)
    with torch.no_grad():
        # 거리커널: 평균 타입쌍에서 P(연결) vs 거리
        ds = torch.linspace(0.05, 3.0, 60, device=DEV)
        avg_t = EYE.mean(0, keepdim=True)
        kern = torch.sigmoid(model(avg_t.expand(60, T), avg_t.expand(60, T), ds[:, None])).cpu().numpy()
        # 타입친화도: 중간거리에서 (ti,tj) P(연결)
        med = torch.full((T * T, 1), 0.5, device=DEV)
        ii, jj = torch.meshgrid(torch.arange(T, device=DEV), torch.arange(T, device=DEV), indexing="ij")
        aff = torch.sigmoid(model(EYE[ii.reshape(-1)], EYE[jj.reshape(-1)], med)).cpu().numpy().reshape(T, T)

    # --- 큰 뇌 생성 (공간-국소 희소) ---
    print(f"\n{N_BIG:,}뉴런 ({N_BIG/N:.0f}배) 생성 중...")
    t0 = time.time()
    side = (N_BIG / dvol) ** (1 / 3)
    bpos = torch.rand(N_BIG, 3, device=DEV) * side
    btypes = torch.tensor(np.random.choice(T, N_BIG, p=type_dist), device=DEV)
    btoh = EYE[btypes]
    per_cell = 300; cs = (per_cell / dvol) ** (1 / 3); G = int(side / cs) + 2
    cell = (bpos / cs).floor().long().clamp(0, G - 1)
    cid = cell[:, 0] * G * G + cell[:, 1] * G + cell[:, 2]
    order = torch.argsort(cid); sc = cid[order]
    uniq, inv, cnt = torch.unique(sc, return_inverse=True, return_counts=True)
    starts = torch.cumsum(cnt, 0) - cnt
    C = 96
    src_all, dst_all = [], []
    for p0 in range(0, N_BIG, 20000):
        ps = torch.arange(p0, min(p0 + 20000, N_BIG), device=DEV)
        st = starts[inv[ps]]; ln = cnt[inv[ps]].clamp(min=1)
        off = (torch.rand(len(ps), C, device=DEV) * ln.unsqueeze(1)).long()
        cand = order[(st.unsqueeze(1) + off).reshape(-1)]
        si = order[ps].repeat_interleave(C)
        dist = (bpos[si] - bpos[cand]).norm(dim=-1, keepdim=True) / dscale
        with torch.no_grad():
            p = torch.sigmoid(model(btoh[si], btoh[cand], dist)).view(len(ps), C)
        p = p * (mean_deg / C) / p.mean().clamp(min=1e-6)
        fire = torch.rand_like(p) < p
        rows, cols = fire.nonzero(as_tuple=True)
        src_all.append(order[ps][rows]); dst_all.append(cand.view(len(ps), C)[rows, cols])
    src = torch.cat(src_all); dst = torch.cat(dst_all)
    m = src != dst; src, dst = src[m], dst[m]
    print(f"  생성완료: {len(src):,}엣지 (평균차수 {len(src)/N_BIG:.1f}), {time.time()-t0:.1f}초")

    # --- 큰 뇌 구조 분석 ---
    def typeflow_edges(s, dd, t, T):
        M = torch.zeros(T, T, device=DEV)
        M.index_put_((t[s], t[dd]), torch.ones(len(s), device=DEV), accumulate=True)
        return (M / M.sum().clamp(min=1)).flatten()
    Mreal = typeflow_edges(Et[:, 0], Et[:, 1], torch.tensor(nt, device=DEV), T)
    Mbig = typeflow_edges(src, dst, btypes, T)
    a = Mbig - Mbig.mean(); b = Mreal - Mreal.mean()
    tf_corr = float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-8))
    bdeg = torch.zeros(N_BIG, device=DEV).index_add_(0, src, torch.ones(len(src), device=DEV))
    print(f"\n=== 큰 뇌(50k) 구조가 실제 쥐 규칙 유지하나 ===")
    print(f"  세포타입 흐름 상관(실제 쥐와): {tf_corr:.3f}")
    print(f"  평균차수 {bdeg.mean():.1f} (실제 {mean_deg:.1f}) | 최대차수 {int(bdeg.max())}")
    conn_dist = ((bpos[src] - bpos[dst]).norm(dim=-1) / dscale).cpu().numpy()
    print(f"  연결거리 중앙값 {np.median(conn_dist):.2f} (거리스케일 단위) → 국소 연결 유지")

    # --- 시각화 ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].plot(ds.cpu().numpy(), kern, lw=2, color="tab:blue")
    ax[0].set_title("LEARNED RULE 1: P(connect) vs distance"); ax[0].set_xlabel("distance (norm)"); ax[0].set_ylabel("P")
    im = ax[1].imshow(aff, cmap="magma"); ax[1].set_title("LEARNED RULE 2: cell-type affinity")
    ax[1].set_xticks(range(T)); ax[1].set_xticklabels(vocab, rotation=90, fontsize=6)
    ax[1].set_yticks(range(T)); ax[1].set_yticklabels(vocab, fontsize=6); plt.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].hist(bdeg.cpu().numpy(), bins=50, color="tab:red", alpha=0.8)
    ax[2].set_title(f"Generated BIG brain ({N_BIG//1000}k neurons)\ndegree distribution"); ax[2].set_xlabel("degree")
    plt.tight_layout(); plt.savefig(f"{VIZ}/6_bigger_brain_rules.png", dpi=110); plt.close()
    print(f"\n시각화: {VIZ}/6_bigger_brain_rules.png")
    np.savez_compressed(os.path.join(DOUT, "phase12_bigbrain.npz"),
                        kernel=kern, affinity=aff, tf_corr=tf_corr, n=N_BIG, edges=len(src))


if __name__ == "__main__":
    main()
