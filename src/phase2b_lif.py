"""
NOMS-LAB connectome-gen : Phase 2B-1 — GPU LIF 스파이킹 시뮬레이터

A(선형확산)보다 진짜 생물물리에 가깝게: leaky integrate-and-fire 스파이킹.
Shiu 2024(초파리 전뇌 LIF) 축소판. 의존성 없이 PyTorch GPU.

  - 뉴런: LIF  dV = (-V/tau)dt + I_syn + I_ext,  V>=θ 면 스파이크 후 reset+불응기
  - 시냅스: connectome 가중(여기선 이진) × 신경전달물질 부호(Dale: 뉴런별 +/-)
  - 감각뉴런에 외부전류 주입 → 스파이크 전파 → 운동뉴런 발화율 측정
  - 실제 vs 생성 vs 무작위(ER) 발화패턴 비교 (동일 시뮬/동일 부호)

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

# --- LIF 파라미터 ---
DT, TAU, THETA = 0.5, 20.0, 1.0       # ms
T_MS = 600.0                           # 시뮬 길이
REFRAC = 2.0                           # 불응기 ms
G_SYN = 1.6                            # 시냅스 강도
I_SENS = 1.3                           # 감각 외부전류
INH_FRAC = 0.2                         # 억제성 뉴런 비율(Dale)


def lif_sim(A, sign, sens_mask, steps):
    """A: (k,k) i->j 이진. sign: (k,) +1/-1. 반환: 뉴런별 스파이크수."""
    A = torch.tensor(A, dtype=torch.float32, device=DEV)
    sgn = torch.tensor(sign, dtype=torch.float32, device=DEV)
    At = A.t()                                  # At[i,j]=A[j,i] (j pre→i post)
    indeg = At.sum(1, keepdim=True); indeg[indeg == 0] = 1.0
    M = (At * sgn[None, :]) / indeg             # 들어오는 부호가중 정규화
    Iext = torch.tensor(sens_mask, dtype=torch.float32, device=DEV) * I_SENS
    k = A.shape[0]
    V = torch.zeros(k, device=DEV)
    spk = torch.zeros(k, device=DEV)
    refr = torch.zeros(k, device=DEV)
    count = torch.zeros(k, device=DEV)
    leak = DT / TAU
    for _ in range(steps):
        Isyn = G_SYN * (M @ spk)
        V = V * (1 - leak) + Isyn + Iext
        V = torch.where(refr > 0, torch.zeros_like(V), V)   # 불응기 중 정지
        spk = (V >= THETA).float()
        count += spk
        V = torch.where(spk > 0, torch.zeros_like(V), V)    # reset
        refr = torch.where(spk > 0, torch.full_like(refr, REFRAC), torch.clamp(refr - DT, min=0))
    return count.cpu().numpy()


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def main():
    d = np.load(GEN, allow_pickle=True)
    A_real = d["A_real"].astype(np.float32)
    prob = d["prob"].astype(np.float32)
    tix = d["types_idx"]
    k = A_real.shape[0]
    sens = (tix == TIDX["sensory"]).astype(np.float32)
    motor = (tix == TIDX["motor"])
    steps = int(T_MS / DT)

    # Dale 부호: 뉴런별 고정 (실제·생성·무작위 동일하게)
    sign = np.ones(k, np.float32)
    inh = np.random.rand(k) < INH_FRAC
    sign[inh] = -1.0

    real_c = lif_sim(A_real, sign, sens, steps)

    n_samp = 20
    gen_cs, rand_cs = [], []
    target_edges = int(A_real.sum())
    for _ in range(n_samp):
        Ag = (np.random.rand(k, k) < prob).astype(np.float32); np.fill_diagonal(Ag, 0)
        gen_cs.append(lif_sim(Ag, sign, sens, steps))
        Ar = np.zeros(k * k, np.float32)
        Ar[np.random.choice(k * k, target_edges, replace=False)] = 1
        Ar = Ar.reshape(k, k); np.fill_diagonal(Ar, 0)
        rand_cs.append(lif_sim(Ar, sign, sens, steps))
    gen_c = np.mean(gen_cs, axis=0); rand_c = np.mean(rand_cs, axis=0)

    # 발화율 (Hz) = 스파이크수 / 시뮬초
    to_hz = 1000.0 / T_MS
    print(f"노드 {k} | 감각 {int(sens.sum())} | 운동 {int(motor.sum())} | 억제성 {int(inh.sum())}")
    print(f"\n=== LIF 평균 발화율 (Hz) ===")
    print(f"{'':<10}{'전체':>8}{'운동':>8}")
    for name, c in [("실제", real_c), ("생성", gen_c), ("무작위", rand_c)]:
        print(f"{name:<10}{c.mean()*to_hz:>8.1f}{c[motor].mean()*to_hz:>8.1f}")

    cg = pearson(real_c[motor], gen_c[motor]); cr = pearson(real_c[motor], rand_c[motor])
    cga = pearson(real_c, gen_c); cra = pearson(real_c, rand_c)
    print(f"\n--- 운동뉴런 발화패턴 상관 (실제와) ---")
    print(f"  생성  vs 실제 : {cg:.3f}   (전체뉴런 {cga:.3f})")
    print(f"  무작위 vs 실제 : {cr:.3f}   (전체뉴런 {cra:.3f})")
    print(f"  판정: {'생성 >> 무작위 (기능 재현 O)' if cg > cr + 0.1 else '생성 ~ 무작위'}")

    np.savez_compressed(os.path.join(OUTPUTS, "phase2b_lif.npz"),
                        real_c=real_c, gen_c=gen_c, rand_c=rand_c, sign=sign,
                        motor_mask=motor, sens_mask=sens)
    print(f"\n저장: {os.path.join(OUTPUTS, 'phase2b_lif.npz')}")


if __name__ == "__main__":
    main()
