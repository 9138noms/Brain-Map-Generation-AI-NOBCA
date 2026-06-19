"""
NOMS-LAB connectome-gen : Phase 3B — 강건성/일반화 검증

단계3 약점: 단일 자극조건 최적화 → 과적합 의심.
검증: 여러 자극조건으로 최적화 → *학습 안 한* 조건에서도 기능 재현되나(일반화)?

  - 자극조건 K개 = 감각뉴런의 서로 다른 활성 패턴 (서로 다른 "자극")
  - train 조건으로만 생성기(logits) 최적화 (RL, 샘플효율)
  - test(held-out) 조건에서 기능상관 측정: v2 / 최적화 / 무작위 비교
  - test에서도 최적화 > v2 > 무작위 면 = 과적합 아님, 진짜 기능 재현
"""
import os, sys
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = r"D:\NOMS-LAB-D\connectome-gen\outputs"
GEN = os.path.join(OUT, "phase1_v2_generated_stage8.npz")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TIDX = {"sensory": 0, "inter": 1, "motor": 2, "modulatory": 3, "glia": 4, "other": 5}
torch.manual_seed(0); np.random.seed(0)

DT, TAU, THETA, REFRAC = 0.5, 20.0, 1.0, 2.0
STEPS, G_SYN, I_SENS, INH = 500, 1.6, 1.3, 0.2
K_COND, N_TRAIN = 12, 8
N_SAMP, ITERS, LR = 12, 140, 0.05
LAM_DENS, LAM_KL = 30.0, 0.02


def lif_batch(A, sign, drive):
    """drive: (k,) 조건별 자극 벡터."""
    N, k, _ = A.shape
    At = A.transpose(1, 2)
    indeg = At.sum(2, keepdim=True).clamp(min=1.0)
    M = (At * sign.view(1, 1, k)) / indeg
    Iext = drive.view(1, k)
    V = torch.zeros(N, k, device=DEV); spk = torch.zeros(N, k, device=DEV)
    refr = torch.zeros(N, k, device=DEV); cnt = torch.zeros(N, k, device=DEV)
    leak = DT / TAU
    for _ in range(STEPS):
        Isyn = G_SYN * torch.bmm(M, spk.unsqueeze(-1)).squeeze(-1)
        V = V * (1 - leak) + Isyn + Iext
        V = torch.where(refr > 0, torch.zeros_like(V), V)
        spk = (V >= THETA).float(); cnt += spk
        V = torch.where(spk > 0, torch.zeros_like(V), V)
        refr = torch.where(spk > 0, torch.full_like(refr, REFRAC), (refr - DT).clamp(min=0))
    return cnt


def batch_corr(counts, target):
    c = counts - counts.mean(1, keepdim=True); t = target - target.mean()
    num = (c * t).sum(1); den = c.pow(2).sum(1).sqrt() * t.pow(2).sum().sqrt()
    return num / den.clamp(min=1e-8)


def main():
    d = np.load(GEN, allow_pickle=True)
    A_real = torch.tensor(d["A_real"], dtype=torch.float32, device=DEV)
    prob0 = torch.tensor(d["prob"], dtype=torch.float32, device=DEV).clamp(1e-4, 1 - 1e-4)
    tix = torch.tensor(d["types_idx"], device=DEV); k = A_real.shape[0]
    sens = (tix == TIDX["sensory"]); motor = (tix == TIDX["motor"])
    eye = torch.eye(k, dtype=torch.bool, device=DEV)
    sign = torch.ones(k, device=DEV); sign[torch.rand(k, device=DEV) < INH] = -1.0
    target_dens = A_real.sum() / (k * (k - 1))
    sidx = torch.where(sens)[0]

    # 자극조건 K개: 감각뉴런 부분집합 활성 (서로 다른 자극)
    drives, real_motor = [], []
    for c in range(K_COND):
        on = torch.rand(len(sidx), device=DEV) < 0.5
        dv = torch.zeros(k, device=DEV); dv[sidx[on]] = I_SENS
        drives.append(dv)
        real_motor.append(lif_batch(A_real.unsqueeze(0), sign, dv)[0][motor])
    train_c, test_c = list(range(N_TRAIN)), list(range(N_TRAIN, K_COND))

    def eval_conds(p, conds, n=64):
        cs = []
        with torch.no_grad():
            for c in conds:
                A = (torch.rand(n, k, k, device=DEV) < p).float(); A[:, eye] = 0
                cs.append(batch_corr(lif_batch(A, sign, drives[c])[:, motor], real_motor[c]).mean().item())
        return float(np.mean(cs))

    def overlap(p, n=64):
        with torch.no_grad():
            A = (torch.rand(n, k, k, device=DEV) < p).float(); A[:, eye] = 0
            return ((A * A_real).sum((1, 2)) / A.sum((1, 2)).clamp(min=1)).mean().item()

    p_rand = torch.full((k, k), float(target_dens), device=DEV)   # 무작위 baseline 확률
    base_train = eval_conds(prob0, train_c); base_test = eval_conds(prob0, test_c)
    rand_test = eval_conds(p_rand, test_c)
    print(f"[v2 시작]  train {base_train:.3f}  test {base_test:.3f}  | 무작위 test {rand_test:.3f}")

    L = torch.logit(prob0).clone().detach(); L[eye] = -50.0; L.requires_grad_(True)
    opt = torch.optim.Adam([L], lr=LR)
    baseline = base_train
    for it in range(ITERS):
        p = torch.sigmoid(L)
        loss = 0.0; rmean = []
        for c in train_c:
            with torch.no_grad():
                A = (torch.rand(N_SAMP, k, k, device=DEV) < p).float(); A[:, eye] = 0
                cnt = lif_batch(A, sign, drives[c])
                corr = batch_corr(cnt[:, motor], real_motor[c])
                dens = A.sum((1, 2)) / (k * (k - 1))
                reward = corr - LAM_DENS * (dens - target_dens) ** 2
                rmean.append(reward.mean().item())
            logp = (A * torch.log(p + 1e-8) + (1 - A) * torch.log(1 - p + 1e-8))
            logp = (logp * (~eye).float()).sum((1, 2))
            loss = loss - ((reward - baseline).detach() * logp).mean()
        kl = ((p * torch.log(p / prob0) + (1 - p) * torch.log((1 - p) / (1 - prob0))) * (~eye).float()).sum()
        loss = loss / len(train_c) + LAM_KL * kl
        baseline = 0.9 * baseline + 0.1 * float(np.mean(rmean))
        opt.zero_grad(); loss.backward(); L.grad[eye] = 0; opt.step()
        with torch.no_grad():
            L.data[eye] = -50.0
        if (it + 1) % 35 == 0:
            p = torch.sigmoid(L.detach())
            print(f"  it{it+1:>3}  train {eval_conds(p, train_c):.3f}  test {eval_conds(p, test_c):.3f}")

    p = torch.sigmoid(L.detach())
    fin_train, fin_test, ov = eval_conds(p, train_c), eval_conds(p, test_c), overlap(p)
    print(f"\n=== 강건성 결과 (자극 {K_COND}개 중 train {N_TRAIN} / test {len(test_c)}) ===")
    print(f"{'':<12}{'train조건':>10}{'test조건(미학습)':>16}")
    print(f"{'v2 시작':<12}{base_train:>10.3f}{base_test:>16.3f}")
    print(f"{'최적화후':<12}{fin_train:>10.3f}{fin_test:>16.3f}")
    print(f"{'무작위':<12}{'-':>10}{rand_test:>16.3f}")
    print(f"\n새로움(실제겹침): {ov:.3f}")
    verdict = "일반화 O (과적합 아님)" if fin_test > base_test + 0.05 and fin_test > rand_test + 0.1 else "일반화 약함"
    print(f"판정: {verdict}")
    np.savez_compressed(os.path.join(OUT, "phase3b_robust.npz"), L=L.detach().cpu().numpy())


if __name__ == "__main__":
    main()
