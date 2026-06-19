"""
NOMS-LAB-3 : C. elegans 발달 connectome 로더
Witvliet et al. 2021 (Nature) — 8 발달단계 connectome을 텐서로 변환.

출력 (data/processed/celegans_dev.npz):
  node_names : (N,)   union 뉴런 이름 (정렬)
  node_type  : (N,)   sensory/inter/motor/modulatory/other
  chem       : (S,N,N) 단계별 화학시냅스 가중 인접행렬 (방향성, 시냅스 수)
  gap        : (S,N,N) 단계별 갭정션 인접행렬 (대칭)
  pos        : (S,N,3) 단계별 뉴런 3D 중심좌표 (없으면 nan)
  present    : (S,N)  단계별 뉴런 존재 여부 (bool)
  stage_age  : (S,)   단계 라벨 (1..8)
"""
import json, csv, glob, os, sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # 콘솔 한글 깨짐 방지
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
# 원본 데이터: D (대용량/속도무관) / 가공 출력: C (SSD/속도필요)
ROOT = r"D:\NOMS-LAB-D\connectome-gen\data\raw\nature2021"
NEMA = os.path.join(ROOT, "data", "nemanode")
SKEL = os.path.join(ROOT, "data", "skeletons")
NEURONS_CSV = os.path.join(ROOT, "tables", "_neurons.csv")
OUT = os.path.join(HERE, "..", "data", "processed")  # C: SSD

N_STAGES = 8


def load_edge_lists():
    """8단계 엣지 리스트 로드. typ 0=화학, 2=갭정션."""
    stages = []
    for s in range(1, N_STAGES + 1):
        edges = json.load(open(os.path.join(NEMA, f"witvliet_2020_{s}.json")))
        stages.append(edges)
    return stages


def load_neuron_types():
    """뉴런 class -> type 매핑. 개별 뉴런(ADAL)은 longest-prefix로 class(ADA) 매칭."""
    classes = {}  # class -> type
    with open(NEURONS_CSV) as f:
        for row in csv.DictReader(f):
            classes[row["class"]] = row["type"]
    cls_sorted = sorted(classes, key=len, reverse=True)  # longest first

    def classify(name):
        for c in cls_sorted:
            if name.startswith(c):
                return classes[c]
        return "other"
    return classify


def neuron_centroids(stage_idx):
    """한 단계의 skeleton에서 뉴런별 3D 중심좌표."""
    sk = json.load(open(os.path.join(SKEL, f"Dataset{stage_idx}_skeletons.json")))
    out = {}
    for name, e in sk.items():
        coords = e.get("coords", {})
        if not coords:
            continue
        arr = np.array(list(coords.values()), dtype=float)
        out[name] = arr.mean(axis=0)
    return out


def build():
    stages = load_edge_lists()
    classify = load_neuron_types()

    # union 노드 집합 (모든 단계)
    names = set()
    for edges in stages:
        for e in edges:
            names.add(e["pre"]); names.add(e["post"])
    node_names = sorted(names)
    N = len(node_names)
    idx = {n: i for i, n in enumerate(node_names)}
    node_type = np.array([classify(n) for n in node_names])

    chem = np.zeros((N_STAGES, N, N), dtype=np.float32)
    gap = np.zeros((N_STAGES, N, N), dtype=np.float32)
    pos = np.full((N_STAGES, N, 3), np.nan, dtype=np.float32)
    present = np.zeros((N_STAGES, N), dtype=bool)

    for s, edges in enumerate(stages):
        for e in edges:
            i, j = idx[e["pre"]], idx[e["post"]]
            w = float(sum(e["syn"]))
            present[s, i] = present[s, j] = True
            if e["typ"] == 0:        # 화학 (방향)
                chem[s, i, j] += w
            else:                     # 갭정션 (대칭)
                gap[s, i, j] += w
                gap[s, j, i] += w
        # 좌표
        cents = neuron_centroids(s + 1)
        for n, p in cents.items():
            if n in idx:
                pos[s, idx[n]] = p

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT, "celegans_dev.npz"),
        node_names=np.array(node_names), node_type=node_type,
        chem=chem, gap=gap, pos=pos, present=present,
        stage_age=np.arange(1, N_STAGES + 1),
    )

    # ---- 통계 출력 ----
    print(f"노드(union) : {N}")
    from collections import Counter
    print("타입 분포   :", dict(Counter(node_type)))
    print(f"\n{'단계':>4} {'뉴런':>5} {'화학엣지':>7} {'화학시냅스':>9} {'갭엣지':>6} {'좌표有':>6}")
    for s in range(N_STAGES):
        nseen = present[s].sum()
        ce = (chem[s] > 0).sum()
        csyn = int(chem[s].sum())
        ge = (gap[s] > 0).sum() // 2
        haspos = (~np.isnan(pos[s, :, 0])).sum()
        print(f"{s+1:>4} {nseen:>5} {ce:>7} {csyn:>9} {ge:>6} {haspos:>6}")
    print(f"\n저장: {os.path.join(OUT, 'celegans_dev.npz')}")


if __name__ == "__main__":
    build()
