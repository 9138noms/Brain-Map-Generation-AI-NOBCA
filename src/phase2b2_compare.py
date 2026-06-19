"""
NOMS-LAB connectome-gen : B2 비교 — c302 운동뉴런 활성 상관 (실제 vs 생성 vs 무작위)
B1(우리 LIF)과 동일 기준으로 비교.
"""
import os, sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = r"D:\NOMS-LAB-D\connectome-gen\outputs"


def load(m):
    d = np.load(os.path.join(OUT, f"c302_{m}.npz"), allow_pickle=True)
    return d["names"], d["types"], d["act"].astype(float)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def main():
    names, types, real = load("real")
    _, _, gen = load("gen")
    _, _, rand = load("rand")
    motor = (types == "motor")
    valid = motor & np.isfinite(real) & np.isfinite(gen) & np.isfinite(rand)
    nz = (real[valid].std() > 0)
    print(f"운동뉴런 {motor.sum()} | 유효(활성기록) {valid.sum()}")
    print(f"\n=== c302 운동뉴런 활성(전압 std) ===")
    print(f"{'':<8}{'평균활성':>10}")
    for nm, a in [("실제", real), ("생성", gen), ("무작위", rand)]:
        print(f"{nm:<8}{np.nanmean(a[valid]):>10.4f}")
    cg = pearson(real[valid], gen[valid]); cr = pearson(real[valid], rand[valid])
    print(f"\n--- c302 운동뉴런 활성패턴 상관 (실제와) ---")
    print(f"  생성  vs 실제 : {cg:.3f}")
    print(f"  무작위 vs 실제 : {cr:.3f}")
    print(f"\n=== 작동검증 3종 종합 (생성 vs 실제 상관 / 무작위 vs 실제) ===")
    print(f"  A  선형확산 : 0.379 / -0.168")
    print(f"  B1 GPU LIF  : 0.509 / -0.157")
    print(f"  B2 c302     : {cg:.3f} / {cr:.3f}")


if __name__ == "__main__":
    main()
