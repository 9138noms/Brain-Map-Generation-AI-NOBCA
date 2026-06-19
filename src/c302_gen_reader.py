"""
NOMS-LAB connectome-gen : c302용 커스텀 DataReader

생성/실제 connectome을 OpenWorm c302에 꽂는 어댑터.
환경변수 C302_CONN_MODE = "real" | "gen" 로 어느 connectome 쓸지 선택.

c302 인터페이스:
  read_data(include_nonconnected_cells=False) -> (cell_names, [ConnectionInfo,...])
  read_muscle_data() -> (neurons, muscles, conns)

우리 데이터는 뇌(nerve ring) connectome이라 근육 없음 → 근육 데이터 비움,
운동뉴런 활성으로 비교 (B1과 동일 기준).
"""
import os
import numpy as np
from c302.ConnectomeReader import ConnectionInfo, PREFERRED_NEURON_NAMES

KNOWN = set(PREFERRED_NEURON_NAMES)   # c302가 아는 표준 뉴런만 (근육/시스세포 제외)

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "..", "data", "processed", "celegans_dev.npz")
GEN = r"D:\NOMS-LAB-D\connectome-gen\outputs\phase1_v2_generated_stage8.npz"


def _synclass(pre):
    # c302 관례: D-class 운동뉴런=GABA, 나머지=Acetylcholine (간이 추정)
    return "GABA" if pre.startswith(("DD", "VD")) else "Acetylcholine"


def _load():
    mode = os.environ.get("C302_CONN_MODE", "gen")   # 호출시점에 읽음
    proc = np.load(PROC, allow_pickle=True)
    names_all = [str(x) for x in proc["node_names"]]
    g = np.load(GEN, allow_pickle=True)
    real = g["A_real"].astype(np.float32)
    if mode == "real":
        A = real
    elif mode == "rand":                              # 같은 밀도 무작위 baseline
        A = np.zeros(real.size, np.float32)
        A[np.random.default_rng(0).choice(real.size, int(real.sum()), replace=False)] = 1
        A = A.reshape(real.shape); np.fill_diagonal(A, 0)
    else:
        A = g["A_gen"].astype(np.float32)
    idx = g["node_idx"]
    names = [names_all[i] for i in idx]
    return names, A


def read_data(include_nonconnected_cells=False):
    names, A = _load()
    k = len(names)
    conns = []
    for i in range(k):
        for j in range(k):
            if A[i, j] > 0 and names[i] in KNOWN and names[j] in KNOWN:
                conns.append(ConnectionInfo(names[i], names[j], int(A[i, j]),
                                            "Send", _synclass(names[i])))
    cells = sorted(set([c.pre_cell for c in conns] + [c.post_cell for c in conns]))
    print(f"[c302_gen_reader] MODE={os.environ.get('C302_CONN_MODE','gen')}  cells={len(cells)}  conns={len(conns)}")
    return cells, conns


def read_muscle_data():
    # 근육 없음 (뇌 connectome)
    return [], [], []


if __name__ == "__main__":
    cells, conns = read_data()
    print("sample conns:", [(c.pre_cell, c.post_cell, c.number) for c in conns[:5]])
