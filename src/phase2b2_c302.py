"""
NOMS-LAB connectome-gen : Phase 2B-2 — OpenWorm c302 진짜 작동검증

생성/실제 connectome을 c302(NeuroML)로 빌드 → jNeuroML로 시뮬 → 운동뉴런 활성.
B1(우리 LIF)과 비교용 = "표준 생물물리 도구"에서도 생성 뇌가 작동하나.

사용: py -3.12 src/phase2b2_c302.py <gen|real> [duration_ms] [param_set]
출력: D .../outputs/c302_<mode>.npz  (뉴런이름 -> 활성)
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # c302_gen_reader 임포트용
PROC = os.path.join(HERE, "..", "data", "processed", "celegans_dev.npz")
GEN = r"D:\NOMS-LAB-D\connectome-gen\outputs\phase1_v2_generated_stage8.npz"
OUTDIR = r"D:\NOMS-LAB-D\connectome-gen\outputs"

MODE = sys.argv[1] if len(sys.argv) > 1 else "gen"
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
PSET = sys.argv[3] if len(sys.argv) > 3 else "C"
os.environ["C302_CONN_MODE"] = MODE

# param C 기본=자극0pA(죽음). 살리되 subthreshold(graded) 영역으로 약하게
# → 뉴런별 탈분극 정도가 connectome 라우팅에 따라 달라짐 (C.elegans는 대부분 비-스파이킹)
STIM_PA = float(os.environ.get("C302_STIM", "4"))      # 감각 자극전류 pA
GEXC_NS = float(os.environ.get("C302_GEXC", "0.3"))    # 흥분성 시냅스 전도도 nS
GINH_NS = float(os.environ.get("C302_GINH", "0.3"))    # 억제성 시냅스 전도도 nS

import importlib
import c302
from pyneuroml import pynml

# c302는 data_reader를 항상 "c302.<name>"로 임포트 → 우리 모듈을 그 이름으로 등록
import c302_gen_reader
sys.modules["c302.c302_gen_reader"] = c302_gen_reader


def neuron_names_types():
    proc = np.load(PROC, allow_pickle=True)
    names_all = [str(x) for x in proc["node_names"]]
    types_all = [str(x) for x in proc["node_type"]]
    g = np.load(GEN, allow_pickle=True); idx = g["node_idx"]
    names = [names_all[i] for i in idx]
    types = [types_all[i] for i in idx]
    return names, types


def find_trace(data, name):
    for k in data:
        if k == "t":
            continue
        segs = k.replace("[", "/").replace("]", "/").split("/")
        if name in segs:
            return np.array(data[k])
    for k in data:                              # 폴백: 부분일치
        if k != "t" and name in k:
            return np.array(data[k])
    return None


def main():
    names, types = neuron_names_types()
    sensory = [n for n, t in zip(names, types) if t == "sensory"]
    target = os.path.join(OUTDIR, f"c302_{MODE}")
    os.makedirs(target, exist_ok=True)
    ref = f"c302_{PSET}_{MODE}"

    ParamModel = getattr(importlib.import_module(f"c302.parameters_{PSET}"), "ParameterisedModel")
    params = ParamModel()

    print(f"[B2] MODE={MODE} param={PSET} dur={DUR}ms 뉴런={len(names)} 감각자극={len(sensory)}")
    t0 = time.time()
    overrides = {
        "unphysiological_offset_current": f"{STIM_PA}pA",     # 0pA→자극 살림
        "neuron_to_neuron_chem_exc_syn_gbase": f"{GEXC_NS}nS",
        "neuron_to_neuron_chem_inh_syn_gbase": f"{GINH_NS}nS",
    }
    c302.generate(
        ref, params,
        data_reader="c302_gen_reader",
        cells=None,
        cells_to_plot=names,                    # 전 뉴런 전압 기록
        cells_to_stimulate=sensory,             # 감각뉴런 자극
        muscles_to_include=[],
        duration=DUR, dt=0.05,
        target_directory=target,
        param_overrides=overrides,
        verbose=False,
    )
    print(f"  생성 완료 ({time.time()-t0:.1f}s). LEMS 시뮬 시작...")

    lems = f"LEMS_{ref}.xml"
    t1 = time.time()
    data = pynml.run_lems_with_jneuroml(
        lems, exec_in_dir=target, nogui=True, load_saved_data=True, plot=False, verbose=False)
    print(f"  시뮬 완료 ({time.time()-t1:.1f}s)")

    # 뉴런별 활성 = 평균 탈분극 (graded 영역: connectome 라우팅 반영)
    act = {}
    miss = 0
    for n in names:
        tr = find_trace(data, n)
        act[n] = float(np.mean(tr)) if tr is not None and len(tr) > 1 else np.nan
        if tr is None:
            miss += 1
    print(f"  트레이스 매칭: {len(names)-miss}/{len(names)} (못찾음 {miss})")

    np.savez_compressed(os.path.join(OUTDIR, f"c302_{MODE}.npz"),
                        names=np.array(names), types=np.array(types),
                        act=np.array([act[n] for n in names]))
    print(f"  저장: c302_{MODE}.npz  | 총 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
