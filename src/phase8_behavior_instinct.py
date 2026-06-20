"""
NOMS-LAB connectome-gen : 단계 VIII — 생성 뇌의 행동 + 본능 분석

질문1 (행동): 생성 뇌가 작동할 때 *무슨 행동*을 하나?
질문2 (본능): 생물의 하드와이어 본능이 생성 뇌에도 있나, 아니면 다른 무언가인가?

C. elegans 본능 = 배선 그 자체. 잘 알려진 터치-회피 반사:
  - 앞쪽 터치(ALM/AVM) → 후진 명령뉴런(AVA/AVD/AVE) → 뒤로 도망 (escape)
  - 뒤쪽 터치(PLM)     → 전진 명령뉴런(AVB/PVC) → 앞으로 가속
이 회로를 생성 뇌가 보존하면 = 본능을 물려받음. 학습 없이 자극→적응적 반응.

방법: 각 터치 자극 → LIF → 후진명령 vs 전진명령 활성 = "행동 방향".
      실제 vs 생성 vs 무작위 비교.
"""
import os, sys
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_NPZ = os.path.join(HERE, "..", "data", "processed", "celegans_dev.npz")
GEN = r"D:\NOMS-LAB-D\connectome-gen\outputs\phase1_v2_generated_stage8.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT, TAU, THETA, REFRAC = 0.5, 20.0, 1.0, 2.0
STEPS, G_SYN, I_SENS, INH = 500, 1.6, 1.3, 0.2
np.random.seed(0); torch.manual_seed(0)

# 알려진 회로 (표준 C. elegans 명명)
ANT_TOUCH = ["ALML", "ALMR", "AVM"]          # 앞쪽 부드러운 터치 수용기
POST_TOUCH = ["PLML", "PLMR"]                # 뒤쪽 터치
CMD_BACK = ["AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"]  # 후진 명령
CMD_FWD = ["AVBL", "AVBR", "PVCL", "PVCR"]                   # 전진 명령


def propagate(A, drive, steps=80, alpha=0.9):
    """부호 없는 순수 신호 라우팅 (배선 경로만 측정, 본능=경로 문제이므로)."""
    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=torch.float32, device=DEV)
    At = A.t(); rs = At.sum(1, keepdim=True); rs[rs == 0] = 1.0
    Ahat = At / rs
    x = torch.zeros(A.shape[0], device=DEV)
    for _ in range(steps):
        x = alpha * (Ahat @ x) + drive
    return x


def main():
    d = np.load(DEV_NPZ, allow_pickle=True)
    names_all = [str(x) for x in d["node_names"]]
    g = np.load(GEN, allow_pickle=True)
    idx = g["node_idx"]; prob = torch.tensor(g["prob"], dtype=torch.float32, device=DEV)
    A_real = torch.tensor(g["A_real"], dtype=torch.float32, device=DEV)
    names = [names_all[i] for i in idx]
    loc = {n: i for i, n in enumerate(names)}
    k = len(names); eye = torch.eye(k, dtype=torch.bool, device=DEV)

    def mask(group):
        m = torch.zeros(k, device=DEV)
        present = [n for n in group if n in loc]
        for n in present:
            m[loc[n]] = 1.0
        return m, present

    ant, ant_p = mask(ANT_TOUCH); post, post_p = mask(POST_TOUCH)
    back, back_p = mask(CMD_BACK); fwd, fwd_p = mask(CMD_FWD)
    print(f"회로 뉴런 존재확인: 앞터치{ant_p} 뒤터치{post_p}")
    print(f"  후진명령{back_p}  전진명령{fwd_p}\n")

    sign = torch.ones(k, device=DEV); sign[torch.rand(k, device=DEV) < INH] = -1.0
    # 생성/무작위 샘플
    target = int(A_real.sum().item())
    gens = [((torch.rand(k, k, device=DEV) < prob).float()) for _ in range(12)]
    rnds = []
    for _ in range(12):
        f = torch.zeros(k * k, device=DEV); f[torch.randperm(k * k, device=DEV)[:target]] = 1
        rnds.append(f.view(k, k))
    for a in gens + rnds + [A_real]:
        a[eye] = 0

    def behavior(A, stim):
        x = propagate(A, stim * I_SENS)
        b = (x * back).sum() / max(back.sum(), 1)
        f = (x * fwd).sum() / max(fwd.sum(), 1)
        return float((b - f) / (b + f + 1e-9))   # 정규화 방향성 [-1,1], >0=후진우세

    print("=== 행동: 터치 자극 → 명령뉴런 (후진 - 전진 활성차) ===")
    print(f"{'자극':<14}{'실제':>10}{'생성':>10}{'무작위':>10}{'기대본능':>12}")
    for label, stim, expect in [("앞터치", ant, "후진(회피)"), ("뒤터치", post, "전진(가속)")]:
        r = behavior(A_real, stim)
        gv = float(np.mean([behavior(a, stim) for a in gens]))
        nv = float(np.mean([behavior(a, stim) for a in rnds]))
        print(f"{label:<14}{r:>10.2f}{gv:>10.2f}{nv:>10.2f}{expect:>12}")

    # === 구조적 경로 분석 (부호·동역학 무관, 순수 배선) ===
    # 앞터치 → (1+2홉) → 후진명령 vs 전진명령 경로 비율
    def pathway_ratio(A):
        reach = A + A @ A                       # 1홉 + 2홉
        to_b = (ant @ reach @ back).item()
        to_f = (ant @ reach @ fwd).item()
        return to_b / (to_b + to_f + 1e-9)
    pr_real = pathway_ratio(A_real)
    pr_gen = float(np.mean([pathway_ratio(a) for a in gens]))
    pr_rnd = float(np.mean([pathway_ratio(a) for a in rnds]))
    print(f"\n=== 본능 회로(앞터치→명령) 구조적 경로비율 [후진/(후진+전진)] ===")
    print(f"  실제   : {pr_real:.3f}   (0.5=중립, >0.5=후진(회피)회로 우세)")
    print(f"  생성   : {pr_gen:.3f}")
    print(f"  무작위 : {pr_rnd:.3f}")
    # 본능 보존 = 생성이 실제의 편향 방향을 따라가나
    follows = abs(pr_gen - pr_real) < abs(pr_rnd - pr_real)
    print(f"\n=== 판정 ===")
    print(f"  생성이 실제 편향을 무작위보다 잘 따라감: {follows}")
    if follows and abs(pr_real - 0.5) > 0.03:
        print(f"  → 생성 뇌가 본능 회로의 *편향 방향*을 부분 보존 (정확한 반사는 부호 필요).")
    else:
        print(f"  → 특정 본능 회로는 보존 약함 — 통계적 생성이라 정밀 반사배선은 안 따라옴.")


if __name__ == "__main__":
    main()
