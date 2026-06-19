"""
NOMS-LAB connectome-gen : Phase 3 (ES) — 기능 최적화를 진화전략으로

RL(REINFORCE) 대신 OpenAI-ES(antithetic, rank-normalized)로 같은 목적 최적화.
생물 도메인엔 진화전략이 자연스럽고 보통 더 견고. RL 버전과 직접 비교용.

목적(동일): 엣지 logits L 을 최적화 → 샘플 connectome의 LIF 기능상관 최대화
            + 밀도 고정 + v2 사전 KL. 새로움(실제겹침) 모니터.
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
STEPS, G_SYN, I_SENS, INH = 600, 1.6, 1.3, 0.2
# ES 하이퍼파라미터 (고차원 45k → 인구 대폭 확대 + Adam)
POP, GENS, SIGMA, LR = 256, 300, 0.3, 0.1
LAM_DENS, LAM_KL = 30.0, 0.02


def lif_batch(A, sign, sens):
    N, k, _ = A.shape
    At = A.transpose(1, 2)
    indeg = At.sum(2, keepdim=True).clamp(min=1.0)
    M = (At * sign.view(1, 1, k)) / indeg
    Iext = (sens * I_SENS).view(1, k)
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


@torch.no_grad()
def main():
    d = np.load(GEN, allow_pickle=True)
    A_real = torch.tensor(d["A_real"], dtype=torch.float32, device=DEV)
    prob0 = torch.tensor(d["prob"], dtype=torch.float32, device=DEV).clamp(1e-4, 1 - 1e-4)
    tix = torch.tensor(d["types_idx"], device=DEV)
    k = A_real.shape[0]
    sens = (tix == TIDX["sensory"]).float(); motor = (tix == TIDX["motor"])
    eye = torch.eye(k, dtype=torch.bool, device=DEV)
    sign = torch.ones(k, device=DEV); sign[torch.rand(k, device=DEV) < INH] = -1.0
    target_dens = A_real.sum() / (k * (k - 1))
    real_motor = lif_batch(A_real.unsqueeze(0), sign, sens)[0][motor]
    logprob0 = torch.log(prob0); log1m0 = torch.log(1 - prob0)
    maskf = (~eye).float()

    def fitness(logits):                       # logits:(P,k,k) -> (P,)
        logits = logits.clone(); logits[:, eye] = -50.0
        p = torch.sigmoid(logits)
        A = (torch.rand_like(p) < p).float(); A[:, eye] = 0
        cnt = lif_batch(A, sign, sens)
        corr = batch_corr(cnt[:, motor], real_motor)
        dens = A.sum((1, 2)) / (k * (k - 1))
        kl = ((p * (torch.log(p) - logprob0) + (1 - p) * (torch.log(1 - p) - log1m0)) * maskf).sum((1, 2))
        return corr - LAM_DENS * (dens - target_dens) ** 2 - LAM_KL * kl

    def evaluate(theta, n=128):
        logits = theta.view(1, k, k).expand(n, k, k).clone(); logits[:, eye] = -50.0
        p = torch.sigmoid(logits); A = (torch.rand_like(p) < p).float(); A[:, eye] = 0
        cnt = lif_batch(A, sign, sens)
        corr = batch_corr(cnt[:, motor], real_motor).mean().item()
        dens = (A.sum((1, 2)) / (k * (k - 1))).mean().item()
        ov = ((A * A_real).sum((1, 2)) / A.sum((1, 2)).clamp(min=1)).mean().item()
        return corr, dens, ov

    theta = torch.logit(prob0).flatten().clone()
    P = theta.numel()
    c0, d0, o0 = evaluate(theta)
    print(f"[시작 v2]  기능상관 {c0:.3f}  밀도 {d0:.3f}  실제겹침 {o0:.3f}")

    half = POP // 2
    m = torch.zeros_like(theta); v = torch.zeros_like(theta)   # manual Adam
    b1, b2 = 0.9, 0.999
    for g in range(GENS):
        eps = torch.randn(half, P, device=DEV)
        eps = torch.cat([eps, -eps], 0)                       # antithetic (POP,P)
        cand = (theta[None] + SIGMA * eps).view(POP, k, k)
        fit = fitness(cand)                                   # (POP,)
        ranks = fit.argsort().argsort().float()
        util = ranks / (POP - 1) - 0.5                        # rank 정규화
        grad = (util[:, None] * eps).sum(0) / (POP * SIGMA)   # ascent 방향
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad * grad
        mh = m / (1 - b1 ** (g + 1)); vh = v / (1 - b2 ** (g + 1))
        theta = theta + LR * mh / (vh.sqrt() + 1e-8)
        if (g + 1) % 25 == 0:
            c, dn, ov = evaluate(theta)
            print(f"  gen{g+1:>3}  기능상관 {c:.3f}  밀도 {dn:.3f}  실제겹침 {ov:.3f}")

    cF, dF, oF = evaluate(theta, n=128)
    print(f"\n[ES 최적화 후] 기능상관 {cF:.3f}  밀도 {dF:.3f}  실제겹침 {oF:.3f}")
    print(f"  변화: 기능상관 {c0:.3f} → {cF:.3f}  (실제겹침 {o0:.3f} → {oF:.3f})")
    np.savez_compressed(os.path.join(OUT, "phase3_es.npz"), L=theta.view(k, k).cpu().numpy())
    print(f"\n저장: {os.path.join(OUT, 'phase3_es.npz')}")


if __name__ == "__main__":
    main()
