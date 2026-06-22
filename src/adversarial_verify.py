"""
connectome-gen : 적대적 검증 (우리 주장을 깨려는 시도)

각 핵심 주장의 "가장 그럴듯한 반박"을 직접 테스트:
  T1 기능은 *배선* 때문인가 *세포타입* 때문인가? (가장 중요)
     → 같은 뉴런으로 배선만 무작위/뒤섞기 → 기능이 무너지면 = 배선이 진짜 원인
  T2 새로움은 진짜 차이인가 *샘플링 잡음*인가?
     → 생성-실제 vs 생성-생성 vs 실제-실제 겹침 비교
  T3 타입흐름 0.99는 *얼마나 쉬운* 목표인가?
     → 차수+타입비율만 아는 null이 얼마나 나오나
정직한 판정 출력.
"""
import os, sys
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DEVN = os.path.join(HERE, "..", "data", "processed", "celegans_dev.npz")
GEN = r"D:\NOMS-LAB-D\connectome-gen\outputs\phase1_v2_generated_stage8.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TIDX = {"sensory": 0, "inter": 1, "motor": 2, "modulatory": 3, "glia": 4, "other": 5}
np.random.seed(0); torch.manual_seed(0)
DT, TAU, THETA, REFRAC, STEPS, G_SYN, I_SENS, INH = 0.5, 20.0, 1.0, 2.0, 500, 1.6, 1.3, 0.2


def lif(A, sign, sens, motor):
    A = torch.tensor(A, dtype=torch.float32, device=DEV) if not torch.is_tensor(A) else A
    At = A.t(); indeg = At.sum(1, keepdim=True).clamp(min=1); M = (At * sign.view(1, -1)) / indeg
    V = torch.zeros(A.shape[0], device=DEV); spk = torch.zeros_like(V); refr = torch.zeros_like(V); cnt = torch.zeros_like(V)
    drive = sens * I_SENS
    for _ in range(STEPS):
        V = V * (1 - DT / TAU) + G_SYN * (M @ spk) + drive
        V = torch.where(refr > 0, torch.zeros_like(V), V)
        spk = (V >= THETA).float(); cnt += spk
        V = torch.where(spk > 0, torch.zeros_like(V), V)
        refr = torch.where(spk > 0, torch.full_like(refr, REFRAC), (refr - DT).clamp(min=0))
    return cnt


def pear(a, b):
    a = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a, float)
    b = b.detach().cpu().numpy() if torch.is_tensor(b) else np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / (n if n > 1e-9 else 1e-9))


def jac(A, B):
    a = A > 0; b = B > 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def degree_shuffle(A):
    """out-degree 보존 + 무작위 타겟 (배선만 뒤섞기, 같은 뉴런·차수)."""
    N = A.shape[0]; out = (A > 0).sum(1)
    B = np.zeros_like(A)
    for i in range(N):
        k = int(out[i])
        if k:
            t = np.random.choice(N, k, replace=False)
            B[i, t] = 1
    np.fill_diagonal(B, 0)
    return B


def main():
    d = np.load(DEVN, allow_pickle=True); g = np.load(GEN, allow_pickle=True)
    idx = g["node_idx"]; prob = g["prob"]; A_real = (g["A_real"] > 0).astype(float)
    tix = np.array([TIDX.get(str(d["node_type"][i]), 5) for i in idx]); k = len(idx)
    eye = np.eye(k, dtype=bool); sens = (tix == 0); motor = (tix == 2)
    sign = torch.ones(k, device=DEV); sign[torch.rand(k, device=DEV) < INH] = -1
    sm = torch.tensor(sens, dtype=torch.float32, device=DEV); mm = torch.tensor(motor, device=DEV)
    target = int(A_real.sum())

    # ---- T1: 기능은 배선 때문인가? ----
    A_gen = (np.random.rand(k, k) < prob).astype(float); A_gen[eye] = 0
    A_rnd = np.zeros(k * k); A_rnd[np.random.choice(k * k, target, replace=False)] = 1
    A_rnd = A_rnd.reshape(k, k); A_rnd[eye] = 0
    A_dsh = degree_shuffle(A_real)                       # 같은 차수, 배선 뒤섞기
    real_c = lif(A_real, sign, sm, mm)
    def mcorr(A): return pear(real_c[mm], lif(A, sign, sm, mm)[mm])
    print("=== T1: 기능이 배선 때문인가 vs 세포타입 때문인가 (운동뉴런 발화 상관) ===")
    print(f"  생성(학습된 배선)     : {mcorr(A_gen):+.3f}")
    print(f"  무작위 배선(같은 뉴런) : {mcorr(A_rnd):+.3f}")
    print(f"  차수보존 뒤섞기        : {mcorr(A_dsh):+.3f}")
    print(f"  → 무작위/뒤섞기가 0 근처면: 기능은 *배선* 때문 (생성이 진짜 의미있음). 높으면: 껍데기.")

    # ---- T2: 새로움 = 진짜 차이 vs 샘플링 잡음 ----
    gens = [(np.random.rand(k, k) < prob).astype(float) for _ in range(6)]
    for a in gens: a[eye] = 0
    gr = np.mean([jac(a, A_real) for a in gens])
    gg = np.mean([jac(gens[i], gens[j]) for i in range(6) for j in range(i + 1, 6)])
    A7 = (d["chem"][6][np.ix_(idx, idx)] > 0).astype(float); np.fill_diagonal(A7, 0)
    rr = jac(A7, A_real)
    print(f"\n=== T2: 새로움이 진짜 차이인가 샘플링 잡음인가 (겹침 Jaccard) ===")
    print(f"  생성-실제 {gr:.3f} | 생성-생성 {gg:.3f} | 실제-실제(자연변이) {rr:.3f}")
    print(f"  → 생성-실제({gr:.2f}) < 생성-생성({gg:.2f})면: 실제와의 차이가 잡음 이상(체계적). {'통과' if gr < gg - 0.02 else '주의'}")

    # ---- T3: 타입흐름 0.99는 얼마나 쉬운 목표인가 ----
    def tf(A):
        M = np.zeros((6, 6)); s, dd = np.where(A > 0)
        for i, j in zip(tix[s], tix[dd]): M[i, j] += 1
        return (M / max(M.sum(), 1)).flatten()
    Mr = tf(A_real)
    tg = np.mean([pear(tf(a), Mr) for a in gens])
    tr = pear(tf(A_rnd), Mr)
    td = np.mean([pear(tf(degree_shuffle(A_real)), Mr) for _ in range(3)])
    print(f"\n=== T3: 타입흐름 0.99의 난이도 (실제와 상관) ===")
    print(f"  생성 {tg:.3f} | 순수무작위 {tr:.3f} | 차수보존뒤섞기 {td:.3f}")
    print(f"  → 차수보존뒤섞기가 이미 높으면: 타입흐름은 차수+타입이 결정(쉬운 목표). 생성의 우위는 그만큼 약함.")


if __name__ == "__main__":
    main()
