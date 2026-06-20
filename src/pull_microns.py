"""
NOMS-LAB connectome-gen : MICrONS 실제 쥐 피질 connectome 다운로드 → 파이프라인 포맷

minnie65_public (CAVE). 검증된(proofread axon+dendrite) ~2200 뉴런 + 그들 사이 시냅스.
실제 포유류 데이터 → 생성물 사실성 검증의 ground truth.
위치정보(pt_position) 포함 → 공간 배선규칙 사용 가능.

출력: data/processed/pipeline_microns_mouse.npz (num_nodes/node_type/pos/edges/type_vocab)
"""
import os, sys
import numpy as np
from collections import defaultdict
from caveclient import CAVEclient

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "processed", "pipeline_microns_mouse.npz")


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    c = CAVEclient("minnie65_public")
    print("검증 뉴런 조회...")
    pf = c.materialize.query_table("proofreading_status_and_strategy")
    clean = pf[(pf.status_axon == True) & (pf.status_dendrite == True)].copy()
    clean = clean[clean.pt_root_id != 0].drop_duplicates("pt_root_id")
    roots = clean.pt_root_id.tolist()
    rootset = set(roots)
    pos_map = {r: np.array(p, dtype=float) for r, p in zip(clean.pt_root_id, clean.pt_position)}
    print(f"  검증된 뉴런(축삭+수상 둘다): {len(roots)}")

    print("세포타입 조회...")
    ct = c.materialize.query_table("aibs_metamodel_celltypes_v661")
    root2type = {}
    for r, t in zip(ct.pt_root_id, ct.cell_type):
        if r in rootset:
            root2type[r] = str(t)

    print("시냅스 조회 (검증뉴런 사이)...")
    edge_w = defaultdict(int)
    for i, chunk in enumerate(chunks(roots, 400)):
        syn = c.materialize.synapse_query(pre_ids=chunk, post_ids=roots)
        for pre, post in zip(syn["pre_pt_root_id"].values, syn["post_pt_root_id"].values):
            if pre in rootset and post in rootset and pre != post:
                edge_w[(int(pre), int(post))] += 1
        print(f"  청크 {i+1}: 누적 엣지 {len(edge_w)}")

    # 노드 = 시냅스에 등장한 뉴런만 (고립 제외)
    used = set()
    for (a, b) in edge_w:
        used.add(a); used.add(b)
    nodes = [r for r in roots if r in used]
    idx = {r: i for i, r in enumerate(nodes)}
    N = len(nodes)

    types_raw = [root2type.get(r, "unknown") for r in nodes]
    vocab = sorted(set(types_raw))
    tmap = {t: i for i, t in enumerate(vocab)}
    node_type = np.array([tmap[t] for t in types_raw], np.int64)
    pos = np.array([pos_map.get(r, np.zeros(3)) for r in nodes], np.float32)
    edges = np.array([[idx[a], idx[b]] for (a, b) in edge_w], np.int64)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, num_nodes=N, node_type=node_type, pos=pos,
                        edges=edges, type_vocab=np.array(vocab))
    from collections import Counter
    print(f"\n실제 쥐 피질: {N}뉴런 | 엣지 {len(edges):,} | 밀도 {len(edges)/(N*(N-1)):.4f}")
    print(f"타입 {len(vocab)}종:", dict(Counter(types_raw).most_common(10)))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
