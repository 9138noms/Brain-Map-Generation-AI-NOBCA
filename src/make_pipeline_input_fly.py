"""
NOMS-LAB connectome-gen : 초파리 유충 connectome → 파이프라인 포맷 변환기
Winding et al. 2023 (Science) larval Drosophila, 2952뉴런/110k엣지.

벌레와 동일한 표준 포맷(num_nodes/node_type/pos/edges)으로 저장 →
같은 gpu_pipeline.py 가 그대로 학습 (스케일·종 일반성 테스트).
위치정보 없음 → pos=0 (모델은 타입+노드임베딩에 의존).
"""
import os, sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"D:\NOMS-LAB-D\connectome-gen\data\raw\larval-drosophila-connectome\S1\Supplementary-Data-S1"
OUT = os.path.join(HERE, "..", "data", "processed", "pipeline_fly_larva.npz")


def main():
    A = pd.read_csv(os.path.join(BASE, "all-all_connectivity_matrix.csv"), index_col=0)
    assert A.shape[0] == A.shape[1], "정방행렬 아님"
    ids = [str(x) for x in A.index]

    ann = pd.read_csv(os.path.join(BASE, "annotations.csv"))
    id2type = {}
    for _, r in ann.iterrows():
        for col in ("left_id", "right_id"):
            v = str(r[col])
            if v not in ("no pair", "nan", ""):
                id2type[v] = str(r["celltype"])
    types_raw = [id2type.get(i, "other") for i in ids]
    vocab = sorted(set(types_raw))
    tmap = {t: i for i, t in enumerate(vocab)}
    node_type = np.array([tmap[t] for t in types_raw], np.int64)

    M = A.values.astype(np.float32)
    np.fill_diagonal(M, 0)
    ei, ej = np.where(M > 0)
    edges = np.stack([ei, ej], 1).astype(np.int64)
    pos = np.zeros((len(ids), 3), np.float32)   # 위치정보 없음

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, num_nodes=len(ids), node_type=node_type, pos=pos,
                        edges=edges, type_vocab=np.array(vocab))
    from collections import Counter
    print(f"노드 {len(ids)} | 엣지 {len(edges):,} | 밀도 {len(edges)/(len(ids)*(len(ids)-1)):.4f}")
    print(f"타입 {len(vocab)}종:", dict(Counter(types_raw).most_common(8)))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
