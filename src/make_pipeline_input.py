"""
NOMS-LAB connectome-gen : 대규모 파이프라인용 범용 입력포맷 변환기

단일 connectome을 표준 포맷으로 저장 → gpu_pipeline.py가 소비.
여기선 C. elegans 성체(단계8)를 변환. 초파리는 동일 포맷으로 별도 변환기 작성하면 됨.

표준 포맷 (npz):
  num_nodes : int
  node_type : (N,) int  (0=sensory 1=inter 2=motor 3=modulatory 4=glia 5=other)
  pos       : (N,3) float (nan는 평균 대체)
  edges     : (E,2) int  양성 방향엣지 (화학시냅스)
"""
import os, sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "..", "data", "processed", "celegans_dev.npz")
OUT = os.path.join(HERE, "..", "data", "processed", "pipeline_celegans_adult.npz")
TYPES = ["sensory", "inter", "motor", "modulatory", "glia", "other"]
TIDX = {t: i for i, t in enumerate(TYPES)}


def main():
    d = np.load(NPZ, allow_pickle=True)
    s = 7  # 성체
    chem = d["chem"][s]; pos = d["pos"][s]; present = d["present"][s]
    types = d["node_type"]
    ok = present & ~np.isnan(pos[:, 0]); idx = np.where(ok)[0]
    k = len(idx)
    sub = chem[np.ix_(idx, idx)]
    np.fill_diagonal(sub, 0)
    ei, ej = np.where(sub > 0)
    edges = np.stack([ei, ej], 1).astype(np.int64)
    ntype = np.array([TIDX.get(str(types[i]), 5) for i in idx], np.int64)
    P = pos[idx].astype(np.float32)
    # nan 좌표 평균 대체
    col_mean = np.nanmean(P, axis=0)
    inds = np.where(np.isnan(P))
    P[inds] = np.take(col_mean, inds[1])

    np.savez_compressed(OUT, num_nodes=k, node_type=ntype, pos=P, edges=edges)
    print(f"노드 {k} | 양성엣지 {len(edges)} | 밀도 {len(edges)/(k*(k-1)):.4f}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
