"""
NOMS-LAB connectome-gen : 단계 X — 생성 뇌가 실제 뇌처럼 학습 가능한가?

연결망을 순환신경망(reservoir)의 연결구조로 사용 → 그 구조가 제공하는
*계산/기억 능력*(학습의 토대)을 측정. 학습된 readout이 과거입력을 얼마나 복원/계산하나.
  - 선형 기억용량(MC): u(t-k) 복원 R² 합 (얼마나 과거를 기억)
  - 비선형 계산능력: u(t-1)*u(t-3) 복원 R² (기억+비선형 결합 = 진짜 계산)
실제 vs 생성 vs 무작위(같은밀도) 비교. 생성 ≈ 실제 >> 무작위 면 "실제처럼 학습 가능".
"""
import os, sys
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
WORM = r"D:\NOMS-LAB-D\connectome-gen\outputs\phase1_v2_generated_stage8.npz"
MOUSE_IN = os.path.join(HERE, "..", "data", "processed", "pipeline_microns_mouse.npz")
MOUSE_GEN = r"D:\NOMS-LAB-D\connectome-gen\outputs\large\phase9_mouse_gen.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

T, WASH, KMEM, RHO, INSC, RIDGE = 1500, 200, 25, 0.95, 0.5, 1e-3


def reservoir_capacity(mask):
    """mask: (N,N) 0/1 연결구조. 반환 (선형기억용량, 비선형계산R²)."""
    N = mask.shape[0]
    W = torch.randn(N, N, device=DEV) * mask
    # 스펙트럼 반경 RHO로 스케일 (안정적 동역학)
    ev = torch.linalg.eigvals(W).abs().max().real
    W = W * (RHO / ev.clamp(min=1e-6))
    Win = (torch.rand(N, device=DEV) - 0.5) * 2 * INSC
    u = (torch.rand(T, device=DEV) - 0.5) * 1.6                 # 입력 [-0.8,0.8]
    X = torch.zeros(T, N, device=DEV); x = torch.zeros(N, device=DEV)
    for t in range(T):
        x = torch.tanh(W @ x + Win * u[t]); X[t] = x
    Xs = X[WASH:]; us = u[WASH:]
    Xb = torch.cat([Xs, torch.ones(len(Xs), 1, device=DEV)], 1)  # bias
    G = Xb.t() @ Xb + RIDGE * torch.eye(Xb.shape[1], device=DEV)
    Ginv = torch.linalg.inv(G)

    def r2(target):
        w = Ginv @ (Xb.t() @ target)
        pred = Xb @ w
        ss_res = ((target - pred) ** 2).sum(); ss_tot = ((target - target.mean()) ** 2).sum()
        return float((1 - ss_res / ss_tot.clamp(min=1e-9)).clamp(0, 1))

    # 선형 기억용량
    mc = 0.0
    for k in range(1, KMEM + 1):
        tgt = torch.zeros(len(Xs), device=DEV); tgt[k:] = us[:-k]
        mc += r2(tgt)
    # 비선형 계산: u(t-1)*u(t-3)
    nl = torch.zeros(len(Xs), device=DEV)
    nl[3:] = us[2:-1] * us[:-3]
    return mc, r2(nl)


def eval_dataset(name, A_real, prob, reps=3):
    N = A_real.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=DEV)
    dens = A_real.sum() / (N * (N - 1))
    print(f"\n=== {name} ({N}뉴런, 밀도 {dens:.3f}) ===")
    print(f"{'연결구조':<10}{'선형기억용량':>14}{'비선형계산R²':>14}")
    out = {}
    for label in ["실제", "생성", "무작위"]:
        mcs, nls = [], []
        for r in range(reps):
            if label == "실제":
                M = A_real
            elif label == "생성":
                M = (torch.rand(N, N, device=DEV) < prob).float()
            else:
                M = (torch.rand(N, N, device=DEV) < dens).float()
            M = M.clone(); M[eye] = 0
            mc, nl = reservoir_capacity(M)
            mcs.append(mc); nls.append(nl)
        out[label] = (np.mean(mcs), np.mean(nls))
        print(f"{label:<10}{np.mean(mcs):>14.2f}{np.mean(nls):>14.3f}")
    return out


def main():
    # 벌레
    w = np.load(WORM, allow_pickle=True)
    A_real = torch.tensor(w["A_real"], dtype=torch.float32, device=DEV)
    prob = torch.tensor(w["prob"], dtype=torch.float32, device=DEV)
    eval_dataset("벌레 C.elegans", A_real, prob)

    # 쥐
    m = np.load(MOUSE_IN, allow_pickle=True)
    Nm = int(m["num_nodes"]); E = m["edges"]
    Am = torch.zeros(Nm, Nm, device=DEV); Am[torch.tensor(E[:, 0]), torch.tensor(E[:, 1])] = 1
    pgm = torch.tensor(np.load(MOUSE_GEN)["prob"], dtype=torch.float32, device=DEV)
    eval_dataset("쥐 피질 MICrONS", Am, pgm, reps=2)

    print("\n→ 생성 ≈ 실제 >> 무작위 면: 생성 뇌가 실제 뇌처럼 계산/학습 토대를 가짐")


if __name__ == "__main__":
    main()
