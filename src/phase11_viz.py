"""
NOMS-LAB connectome-gen : 단계 XI — 시각화

생성 vs 실제 connectome을 눈으로. 저장: D .../outputs/large/viz/*.png
  1) 벌레 인접행렬 실제 vs 생성 (타입정렬)
  2) 차수분포 (벌레+쥐) 실제/생성/무작위
  3) 쥐 세포타입 흐름행렬 실제 vs 생성
  4) 쥐 뉴런 3D 위치 (타입별 색)
  5) 학습능력 막대 (벌레+쥐)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
WORM = r"D:\NOMS-LAB-D\connectome-gen\outputs\phase1_v2_generated_stage8.npz"
MOUSE_IN = os.path.join(HERE, "..", "data", "processed", "pipeline_microns_mouse.npz")
MOUSE_GEN = r"D:\NOMS-LAB-D\connectome-gen\outputs\large\phase9_mouse_gen.npz"
VIZ = r"D:\NOMS-LAB-D\connectome-gen\outputs\large\viz"
os.makedirs(VIZ, exist_ok=True)


def typeflow(A, t, T):
    M = np.zeros((T, T)); s, d = np.where(A > 0)
    for i, j in zip(t[s], t[d]): M[i, j] += 1
    return M / max(M.sum(), 1)


def main():
    w = np.load(WORM, allow_pickle=True)
    A_real = w["A_real"]; prob = w["prob"]; tix = w["types_idx"]
    A_gen = (np.random.rand(*prob.shape) < prob).astype(float); np.fill_diagonal(A_gen, 0)
    order = np.argsort(tix)

    # 1) 인접행렬
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    ax[0].imshow(A_real[np.ix_(order, order)], cmap="binary", interpolation="nearest")
    ax[0].set_title("Real C. elegans connectome (213)", fontsize=11)
    ax[1].imshow(A_gen[np.ix_(order, order)], cmap="binary", interpolation="nearest")
    ax[1].set_title("AI-generated connectome", fontsize=11)
    for a in ax: a.set_xlabel("post (sorted by type)"); a.set_ylabel("pre")
    plt.tight_layout(); plt.savefig(f"{VIZ}/1_worm_adjacency.png", dpi=110); plt.close()

    # 2) 차수분포
    m = np.load(MOUSE_IN, allow_pickle=True)
    Nm = int(m["num_nodes"]); E = m["edges"]; mt = m["node_type"]
    Am = np.zeros((Nm, Nm)); Am[E[:, 0], E[:, 1]] = 1
    pgm = np.load(MOUSE_GEN)["prob"]
    Agm = (np.random.rand(Nm, Nm) < pgm).astype(float); np.fill_diagonal(Agm, 0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for A, lab, c in [(A_real, "real", "k"), (A_gen, "gen", "tab:red")]:
        deg = A.sum(1); ax[0].hist(deg, bins=30, histtype="step", label=lab, color=c, density=True)
    ax[0].set_title("Worm out-degree"); ax[0].legend(); ax[0].set_xlabel("degree")
    for A, lab, c in [(Am, "real", "k"), (Agm, "gen", "tab:red")]:
        deg = A.sum(1); ax[1].hist(deg, bins=40, histtype="step", label=lab, color=c, density=True)
    ax[1].set_title("Mouse cortex out-degree"); ax[1].legend(); ax[1].set_xlabel("degree")
    plt.tight_layout(); plt.savefig(f"{VIZ}/2_degree_dist.png", dpi=110); plt.close()

    # 3) 쥐 타입흐름
    T = int(mt.max()) + 1
    vocab = [str(x) for x in m["type_vocab"]]
    Mr = typeflow(Am, mt, T); Mg = typeflow(Agm, mt, T)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, M, t in [(ax[0], Mr, "Real mouse type-flow"), (ax[1], Mg, "Generated type-flow")]:
        im = a.imshow(M, cmap="viridis"); a.set_title(t, fontsize=10)
        a.set_xticks(range(T)); a.set_xticklabels(vocab, rotation=90, fontsize=6)
        a.set_yticks(range(T)); a.set_yticklabels(vocab, fontsize=6)
    plt.tight_layout(); plt.savefig(f"{VIZ}/3_mouse_typeflow.png", dpi=110); plt.close()

    # 4) 쥐 3D 위치
    pos = m["pos"]
    fig = plt.figure(figsize=(10, 5))
    for i, (e1, e2, xl, yl) in enumerate([(0, 1, "x", "y"), (0, 2, "x", "z")]):
        a = fig.add_subplot(1, 2, i + 1)
        sc = a.scatter(pos[:, e1], pos[:, e2], c=mt, cmap="tab20", s=4, alpha=0.6)
        a.set_title(f"Mouse neuron positions ({xl}-{yl})", fontsize=10); a.set_aspect("equal")
    plt.tight_layout(); plt.savefig(f"{VIZ}/4_mouse_positions.png", dpi=110); plt.close()

    # 5) 학습능력 막대
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["real", "gen", "random"]
    ax[0].bar(labels, [17.93, 18.35, 21.84], color=["k", "tab:red", "gray"])
    ax[0].set_title("Worm: linear memory capacity")
    ax[1].bar(labels, [0.153, 0.146, 0.190], color=["k", "tab:red", "gray"])
    ax[1].set_title("Worm: nonlinear compute R²")
    plt.tight_layout(); plt.savefig(f"{VIZ}/5_learning_capacity.png", dpi=110); plt.close()

    print(f"저장된 그림 5개: {VIZ}")
    for f in sorted(os.listdir(VIZ)): print("  ", f)


if __name__ == "__main__":
    main()
