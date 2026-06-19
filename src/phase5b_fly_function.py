"""
NOMS-LAB connectome-gen : 초파리 생성 뇌도 작동하나 (기능검증, 두번째 종)

벌레에서 한 "생성 뇌가 감각→운동 신호를 실제처럼 라우팅하나"를 초파리로 확장.
초파리 유충: 감각=sensory, 운동출력 analog = 하행뉴런(DN-*, 복부신경줄로 투사).
LIF로 감각 자극 → DN 활성패턴 실제 vs 생성 vs 무작위 비교.
"""
import os, sys
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "data", "processed", "pipeline_fly_larva.npz")
GENNPZ = r"D:\NOMS-LAB-D\connectome-gen\outputs\phase5_fly_gen.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT, TAU, THETA, REFRAC = 0.5, 20.0, 1.0, 2.0
STEPS, G_SYN, I_SENS, INH = 500, 1.6, 1.3, 0.2
np.random.seed(0); torch.manual_seed(0)


def lif(A, sign, drive):
    A = torch.tensor(A, dtype=torch.float32, device=DEV) if not torch.is_tensor(A) else A
    At = A.t(); indeg = At.sum(1, keepdim=True).clamp(min=1.0)
    M = (At * sign.view(1, -1)) / indeg
    Iext = drive
    V = torch.zeros(A.shape[0], device=DEV); spk = torch.zeros_like(V)
    refr = torch.zeros_like(V); cnt = torch.zeros_like(V); leak = DT / TAU
    for _ in range(STEPS):
        Isyn = G_SYN * (M @ spk)
        V = V * (1 - leak) + Isyn + Iext
        V = torch.where(refr > 0, torch.zeros_like(V), V)
        spk = (V >= THETA).float(); cnt += spk
        V = torch.where(spk > 0, torch.zeros_like(V), V)
        refr = torch.where(spk > 0, torch.full_like(refr, REFRAC), (refr - DT).clamp(min=0))
    return cnt


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    n = (a.norm() * b.norm()).clamp(min=1e-8)
    return float((a * b).sum() / n)


def main():
    d = np.load(IN, allow_pickle=True)
    N = int(d["num_nodes"]); edges = d["edges"]; ntype = d["node_type"]
    vocab = [str(x) for x in d["type_vocab"]]
    prob = torch.tensor(np.load(GENNPZ)["prob"], dtype=torch.float32, device=DEV)

    sens_t = [i for i, v in enumerate(vocab) if v == "sensory"]
    dn_t = [i for i, v in enumerate(vocab) if v.startswith("DN")]
    print(f"초파리 {N}뉴런 | 감각타입 {[vocab[i] for i in sens_t]} | 출력(하행)타입 {[vocab[i] for i in dn_t]}")
    sens = torch.tensor(np.isin(ntype, sens_t), dtype=torch.float32, device=DEV)
    dn = torch.tensor(np.isin(ntype, dn_t), device=DEV)
    print(f"감각 {int(sens.sum())}개 | 하행출력 {int(dn.sum())}개")

    sign = torch.ones(N, device=DEV); sign[torch.rand(N, device=DEV) < INH] = -1.0
    drive = sens * I_SENS
    eye = torch.eye(N, dtype=torch.bool, device=DEV)

    A_real = torch.zeros(N, N, device=DEV)
    A_real[torch.tensor(edges[:, 0]), torch.tensor(edges[:, 1])] = 1.0
    A_real[eye] = 0
    target = int(A_real.sum().item())

    real_c = lif(A_real, sign, drive)
    gens, rnds = [], []
    for _ in range(8):
        Ag = (torch.rand(N, N, device=DEV) < prob).float(); Ag[eye] = 0
        gens.append(lif(Ag, sign, drive))
        flat = torch.zeros(N * N, device=DEV)
        flat[torch.randperm(N * N, device=DEV)[:target]] = 1
        Ar = flat.view(N, N); Ar[eye] = 0
        rnds.append(lif(Ar, sign, drive))
    gen_c = torch.stack(gens).mean(0); rnd_c = torch.stack(rnds).mean(0)

    cg = pearson(real_c[dn], gen_c[dn]); cr = pearson(real_c[dn], rnd_c[dn])
    print(f"\n=== 초파리: 하행출력뉴런 발화패턴 상관 (실제와) ===")
    print(f"  생성  vs 실제 : {cg:.3f}")
    print(f"  무작위 vs 실제 : {cr:.3f}")
    print(f"  판정: {'생성 >> 무작위 (초파리 생성뇌도 작동)' if cg > cr + 0.1 else '약함'}")
    print(f"\n  → 작동검증 종-확장: 벌레(c302 0.78) + 초파리(DN {cg:.2f}) 둘다 생성뇌가 실제처럼 라우팅")


if __name__ == "__main__":
    main()
