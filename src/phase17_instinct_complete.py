"""
NOMS-LAB connectome-gen : 단계 XVII (Tier-A #2) — 본능회로 완성 (가중치 + 부호)

#1 가중치 + 신경전달물질 부호(생물학적)로 escape 반사의 *방향성* 재현 시도.
C. elegans GABA성 뉴런(억제) 알려져 있음 → 부호 부여.
가중치+부호 동역학으로 앞터치 자극 → 후진명령 vs 전진명령 활성.
실제 vs 생성. 생성도 후진 우세(회피)면 = 본능 *기능*까지 재현.
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
WGEN = r"D:\NOMS-LAB-D\connectome-gen\outputs\phase16_weighted.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0); torch.manual_seed(0)

ANT = ["ALML", "ALMR", "AVM"]
CMD_BACK = ["AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"]
CMD_FWD = ["AVBL", "AVBR", "PVCL", "PVCR"]
# C. elegans GABA성(억제) 뉴런 — 머리/회로 관련 (Gendrel 2016 / WormAtlas)
GABA = ["RMED", "RMEV", "RMEL", "RMER", "AVL", "DVB", "RIS",
        "DD1", "DD2", "DD3", "DD4", "DD5", "DD6",
        "VD1", "VD2", "VD3", "VD4", "VD5", "VD6", "VD7", "VD8", "VD9", "VD10", "VD11", "VD12", "VD13"]


def propagate_signed(W, sign, drive, steps=60, alpha=0.9):
    """부호 가중 신호 전파. W[pre,post] 시냅스수, sign[pre] +/-."""
    Wt = (W * sign.view(-1, 1)).t()                      # post×pre, 부호반영
    rs = Wt.abs().sum(1, keepdim=True); rs[rs == 0] = 1.0
    M = Wt / rs
    x = torch.zeros(W.shape[0], device=DEV)
    for _ in range(steps):
        x = torch.tanh(alpha * (M @ x) + drive)
    return x


def main():
    d = np.load(DEVN, allow_pickle=True)
    names_all = [str(x) for x in d["node_names"]]
    g = np.load(GEN, allow_pickle=True); idx = g["node_idx"]
    names = [names_all[i] for i in idx]; loc = {n: i for i, n in enumerate(names)}
    chem = d["chem"][7]
    W_real = torch.tensor(chem[np.ix_(idx, idx)].astype(np.float32), device=DEV)
    W_real.fill_diagonal_(0)
    W_gen = torch.tensor(np.load(WGEN)["W_gen"].astype(np.float32), device=DEV)
    N = len(idx)

    sign = torch.ones(N, device=DEV)
    gaba_present = [n for n in GABA if n in loc]
    for n in gaba_present:
        sign[loc[n]] = -1.0

    def mask(grp):
        m = torch.zeros(N, device=DEV)
        for n in grp:
            if n in loc: m[loc[n]] = 1.0
        return m
    ant, back, fwd = mask(ANT), mask(CMD_BACK), mask(CMD_FWD)
    drive = ant * 1.0

    def escape_bias(W):
        x = propagate_signed(W, sign, drive)
        b = (x * back).sum() / max(back.sum(), 1)
        f = (x * fwd).sum() / max(fwd.sum(), 1)
        return float(b - f)                              # >0 = 후진(회피) 우세

    print(f"GABA성(억제) 뉴런 {len(gaba_present)}개 적용: {gaba_present}")
    print(f"앞터치 {[n for n in ANT if n in loc]}\n")
    r = escape_bias(W_real)
    gs = [escape_bias(W_gen) for _ in range(1)]
    # 무작위 부호 baseline (생물부호 의미 확인용)
    rand_signs = []
    for _ in range(8):
        s2 = torch.ones(N, device=DEV); s2[torch.rand(N, device=DEV) < (len(gaba_present) / N)] = -1
        old = sign.clone(); sign.copy_(s2)
        rand_signs.append(escape_bias(W_real)); sign.copy_(old)

    print(f"=== escape 반사 방향성 (앞터치→ 후진명령 - 전진명령 활성) ===")
    print(f"  실제(가중치+생물부호) : {r:+.3f}   ({'후진=회피 ✓' if r > 0 else '전진'})")
    print(f"  생성(가중치+생물부호) : {np.mean(gs):+.3f}   ({'후진=회피 ✓' if np.mean(gs) > 0 else '전진'})")
    print(f"  실제+무작위부호       : {np.mean(rand_signs):+.3f} (생물부호가 의미있나 대조)")
    print(f"\n→ 실제·생성 둘다 후진(+)이면 가중치+부호로 본능 *기능* 재현")


if __name__ == "__main__":
    main()
