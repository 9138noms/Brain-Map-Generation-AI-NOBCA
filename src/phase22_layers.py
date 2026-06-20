"""
NOMS-LAB connectome-gen : 단계 XXII (Tier-B #7) — 다층/다영역 (피질 층구조)

쥐 피질 = 층 구조(깊이별 세포타입: 23P/4P/5P/6P + 인터뉴런).
깊이축 찾기 → 타입별 깊이 추출 → 층 구조 파라미터 저장(큰뇌 생성용).
층-구조 배치로 생성하면 층간 연결(laminar)이 실제와 같나 데모.
"""
import os, sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
MOUSE = os.path.join(HERE, "..", "data", "processed", "pipeline_microns_mouse.npz")
OUT = os.path.join(HERE, "..", "data", "processed", "mouse_layers.npz")


def main():
    d = np.load(MOUSE, allow_pickle=True)
    pos = d["pos"]; nt = d["node_type"]; vocab = [str(x) for x in d["type_vocab"]]
    T = len(vocab)
    # 깊이축 = 타입별 평균이 가장 잘 분리되는 축 (between-type 분산 최대)
    var_per_axis = []
    for ax in range(3):
        means = [pos[nt == t, ax].mean() for t in range(T) if (nt == t).sum() > 0]
        var_per_axis.append(np.var(means))
    depth_ax = int(np.argmax(var_per_axis))
    depth = pos[:, depth_ax]
    dmin, dmax = depth.min(), depth.max()
    dn = (depth - dmin) / (dmax - dmin + 1e-9)   # 정규화 깊이 0~1

    # 타입별 깊이 평균/표준편차 (층 위치)
    tdepth_mean = np.array([dn[nt == t].mean() if (nt == t).sum() else 0.5 for t in range(T)])
    tdepth_std = np.array([dn[nt == t].std() if (nt == t).sum() else 0.3 for t in range(T)])
    order = np.argsort(tdepth_mean)
    print(f"깊이축 = axis {depth_ax}")
    print("=== 피질 층 구조 (얕음→깊음) ===")
    for t in order:
        if (nt == t).sum() > 0:
            print(f"  {vocab[t]:<10} 깊이 {tdepth_mean[t]:.2f}±{tdepth_std[t]:.2f}  (n={int((nt==t).sum())})")

    # 깊이 구간(10bin)별 타입 분포 → 큰뇌 생성에서 깊이→타입 샘플
    nb = 10
    bin_id = np.clip((dn * nb).astype(int), 0, nb - 1)
    depth_type_prob = np.zeros((nb, T))
    for b in range(nb):
        m = bin_id == b
        if m.sum():
            for t in range(T):
                depth_type_prob[b, t] = (nt[m] == t).mean()
    np.savez(OUT, depth_ax=depth_ax, tdepth_mean=tdepth_mean, tdepth_std=tdepth_std,
             depth_type_prob=depth_type_prob, vocab=np.array(vocab))
    print(f"\n층 파라미터 저장: {OUT}")
    print(f"→ 큰뇌 생성에서 뉴런 깊이 부여 → 깊이별 타입 분포로 층 구조 재현")
    # 검증: 깊이 순서가 알려진 피질 층과 맞나 (23P 얕고 6P 깊어야)
    nm = {vocab[t]: tdepth_mean[t] for t in range(T)}
    checks = [("23P", "6P-CT"), ("23P", "6P-IT"), ("4P", "6P-CT")]
    ok = sum(1 for a, b in checks if a in nm and b in nm and nm[a] < nm[b])
    print(f"층 순서 검증(상층<심층): {ok}/{len(checks)} 일치 " + ("✓ 피질 층구조 포착" if ok >= 2 else ""))


if __name__ == "__main__":
    main()
