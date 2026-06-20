"""
NOMS-LAB connectome-gen : 단계 XXI (Tier-B #6) — 학습하는 뇌

connectome을 순환망의 연결 마스크로 사용, 순환가중치를 역전파로 *학습*.
과제: 지연 회상(입력을 D스텝 뒤 출력) — 기억+계산 필요.
실제 vs 생성 vs 무작위 마스크의 최종 성능 비교. 생성≈실제면 "실제처럼 학습".
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GEN = r"D:\NOMS-LAB-D\connectome-gen\outputs\phase1_v2_generated_stage8.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
T, DELAY, BATCH, ITERS = 40, 5, 64, 400


class MaskedRNN(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.N = mask.shape[0]
        self.mask = mask
        self.W = nn.Parameter(torch.randn(self.N, self.N, device=DEV) * 0.1)
        self.Win = nn.Parameter(torch.randn(self.N, device=DEV) * 0.3)
        self.Wout = nn.Parameter(torch.randn(self.N, device=DEV) * 0.1)
        self.b = nn.Parameter(torch.zeros(1, device=DEV))

    def forward(self, u):  # u: (B,T)
        B = u.shape[0]; h = torch.zeros(B, self.N, device=DEV)
        Wm = self.W * self.mask
        outs = []
        for t in range(u.shape[1]):
            h = torch.tanh(h @ Wm.t() + u[:, t:t+1] * self.Win)
            outs.append(h @ self.Wout + self.b)
        return torch.stack(outs, 1)  # (B,T)


def run(mask, reps=3):
    r2s = []
    for rep in range(reps):
        torch.manual_seed(rep)
        m = MaskedRNN(mask).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        for it in range(ITERS):
            u = (torch.rand(BATCH, T, device=DEV) - 0.5) * 2
            tgt = torch.zeros_like(u); tgt[:, DELAY:] = u[:, :-DELAY]
            pred = m(u)
            loss = ((pred[:, DELAY:] - tgt[:, DELAY:]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            u = (torch.rand(256, T, device=DEV) - 0.5) * 2
            tgt = torch.zeros_like(u); tgt[:, DELAY:] = u[:, :-DELAY]
            pred = m(u)
            res = ((pred[:, DELAY:] - tgt[:, DELAY:]) ** 2).mean()
            var = tgt[:, DELAY:].var()
            r2s.append(float((1 - res / var).clamp(-1, 1)))
    return np.mean(r2s)


def main():
    g = np.load(GEN, allow_pickle=True)
    A_real = torch.tensor((g["A_real"] > 0).astype(np.float32), device=DEV)
    prob = torch.tensor(g["prob"], dtype=torch.float32, device=DEV)
    N = A_real.shape[0]; eye = torch.eye(N, dtype=torch.bool, device=DEV)
    dens = A_real.sum() / (N * (N - 1))
    A_gen = (torch.rand(N, N, device=DEV) < prob).float(); A_gen[eye] = 0
    A_rnd = (torch.rand(N, N, device=DEV) < dens).float(); A_rnd[eye] = 0

    print(f"=== 학습하는 뇌: 지연{DELAY}스텝 회상 과제 (R², 높을수록 학습 잘됨) ===")
    print(f"  실제   : {run(A_real):.3f}")
    print(f"  생성   : {run(A_gen):.3f}")
    print(f"  무작위 : {run(A_rnd):.3f}")
    print("→ 생성≈실제면 생성 뇌가 실제처럼 학습/계산 가능")


if __name__ == "__main__":
    main()
