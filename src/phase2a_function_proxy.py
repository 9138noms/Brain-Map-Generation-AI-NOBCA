"""
NOMS-LAB connectome-gen : Phase 2A — 싼 작동검증 프록시

질문: 생성한 뇌가 구조뿐 아니라 *기능적으로도* 실제처럼 동작하나?
방법: 감각뉴런에 신호 주입 → 선형 확산 전파 → 운동뉴런 활성 패턴 측정.
핵심 지표:
  - 도달성/지연: 감각→운동 경로가 닿나, 몇 홉(BFS)
  - **per-운동뉴런 활성 상관**: 같은 자극에 *같은 운동뉴런*이 켜지나 (실제 vs 생성)
  - 무작위 그래프(ER, 같은 밀도) baseline 대비 — 생성이 우연보다 나은가

주의: 신경전달물질 부호 미반영(전부 흥분성 가정) — Shiu 2024 수준 생물물리는 B(c302)에서.
입력: D .../outputs/phase1_v2_generated_stage8.npz
"""
import os, sys
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUTPUTS = r"D:\NOMS-LAB-D\connectome-gen\outputs"
GEN = os.path.join(OUTPUTS, "phase1_v2_generated_stage8.npz")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TIDX = {"sensory": 0, "inter": 1, "motor": 2, "modulatory": 3, "glia": 4, "other": 5}
np.random.seed(0); torch.manual_seed(0)


def propagate(A, sensory_mask, steps=100, alpha=0.9):
    """선형 확산: x_{t+1} = alpha * (incoming-normalized A^T) x_t + 감각drive."""
    A = torch.tensor(A, dtype=torch.float32, device=DEV)
    At = A.t().clone()                       # At[j,i] = A[i,j] (j로 들어오는)
    rs = At.sum(1, keepdim=True); rs[rs == 0] = 1.0
    Ahat = At / rs                            # 들어오는 가중치 정규화
    drive = torch.tensor(sensory_mask, dtype=torch.float32, device=DEV)
    x = torch.zeros(A.shape[0], device=DEV)
    for _ in range(steps):
        x = alpha * (Ahat @ x) + drive
    return x.cpu().numpy()


def bfs_dist(A, src_mask):
    """감각집합에서 각 노드까지 방향 BFS 홉수 (못 닿으면 inf)."""
    k = A.shape[0]; dist = np.full(k, np.inf)
    frontier = src_mask.astype(bool).copy(); dist[frontier] = 0
    visited = frontier.copy(); t = 0
    while frontier.any():
        t += 1
        reach = (A.T @ frontier.astype(np.float32)) > 0   # frontier에서 1홉
        new = reach & ~visited
        dist[new] = t; visited |= new; frontier = new
    return dist


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def main():
    d = np.load(GEN, allow_pickle=True)
    A_real = d["A_real"].astype(np.float32)
    prob = d["prob"].astype(np.float32)        # 생성 확률행렬 (기대 connectome)
    tix = d["types_idx"]
    k = A_real.shape[0]
    sens = (tix == TIDX["sensory"]); motor = (tix == TIDX["motor"])
    print(f"노드 {k} | 감각 {sens.sum()} | 운동 {motor.sum()}")

    # --- 생성: prob에서 이진 그래프 N개 샘플, 활성 평균 ---
    n_samp = 30
    target_edges = int(A_real.sum())
    gen_acts = []
    A_gens = []
    for _ in range(n_samp):
        Ag = (np.random.rand(k, k) < prob).astype(np.float32); np.fill_diagonal(Ag, 0)
        A_gens.append(Ag)
        gen_acts.append(propagate(Ag, sens))
    gen_act = np.mean(gen_acts, axis=0)
    A_gen_mean_edges = np.mean([a.sum() for a in A_gens])

    # --- 무작위 ER baseline (같은 엣지수) ---
    rand_acts = []
    for _ in range(n_samp):
        Ar = np.zeros(k * k, np.float32)
        Ar[np.random.choice(k * k, target_edges, replace=False)] = 1
        Ar = Ar.reshape(k, k); np.fill_diagonal(Ar, 0)
        rand_acts.append(propagate(Ar, sens))
    rand_act = np.mean(rand_acts, axis=0)

    real_act = propagate(A_real, sens)

    # --- 도달성/지연 (BFS) ---
    dr = bfs_dist(A_real, sens); dg = bfs_dist(A_gens[0], sens)
    def reach_lat(dist):
        md = dist[motor]; fin = md[np.isfinite(md)]
        return md[np.isfinite(md)].size / max(motor.sum(), 1), (fin.mean() if fin.size else float("nan"))
    rr, lr = reach_lat(dr); rg, lg2 = reach_lat(dg)

    # --- 핵심: per-운동뉴런 활성 상관 ---
    cor_gen = pearson(real_act[motor], gen_act[motor])
    cor_rand = pearson(real_act[motor], rand_act[motor])
    cor_gen_all = pearson(real_act, gen_act)
    cor_rand_all = pearson(real_act, rand_act)

    print(f"\n=== 작동검증 프록시 (감각자극 → 운동활성) ===")
    print(f"{'지표':<26}{'실제':>9}{'생성':>9}{'무작위':>9}")
    print(f"{'운동 도달비율':<24}{rr:>9.2f}{rg:>9.2f}{'-':>9}")
    print(f"{'감각→운동 평균홉':<23}{lr:>9.2f}{lg2:>9.2f}{'-':>9}")
    print(f"{'운동 평균활성':<24}{real_act[motor].mean():>9.3f}{gen_act[motor].mean():>9.3f}{rand_act[motor].mean():>9.3f}")
    print(f"\n--- 운동뉴런 활성패턴 상관 (실제와 얼마나 같나) ---")
    print(f"  생성  vs 실제 : {cor_gen:.3f}   (전체뉴런 {cor_gen_all:.3f})")
    print(f"  무작위 vs 실제 : {cor_rand:.3f}   (전체뉴런 {cor_rand_all:.3f})")
    verdict = "생성 >> 무작위 (기능 재현 O)" if cor_gen > cor_rand + 0.1 else "생성 ~ 무작위 (기능 미흡)"
    print(f"  판정: {verdict}")

    np.savez_compressed(os.path.join(OUTPUTS, "phase2a_function.npz"),
                        real_act=real_act, gen_act=gen_act, rand_act=rand_act,
                        motor_mask=motor, sens_mask=sens)
    print(f"\n저장: {os.path.join(OUTPUTS, 'phase2a_function.npz')}")


if __name__ == "__main__":
    main()
