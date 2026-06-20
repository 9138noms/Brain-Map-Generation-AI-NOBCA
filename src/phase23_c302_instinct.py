"""
NOMS-LAB connectome-gen : 단계 XXIII (Tier-A #4) — c302 본능 기능검증

우리 단순 동역학이 못 본 escape 반사를, c302 생물물리(신경전달물질 부호 내장)로 재시도.
앞터치 뉴런만 자극 → 후진명령(AVA/AVD/AVE) vs 전진명령(AVB/PVC) 활성. 실제 vs 생성.
바운드 시도 — 결과 그대로 기록.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROC = os.path.join(HERE, "..", "data", "processed", "celegans_dev.npz")
GEN = r"D:\NOMS-LAB-D\connectome-gen\outputs\phase1_v2_generated_stage8.npz"
OUTDIR = r"D:\NOMS-LAB-D\connectome-gen\outputs"
DUR, PSET = 200.0, "C"

ANT = ["ALML", "ALMR", "AVM"]
BACK = ["AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"]
FWD = ["AVBL", "AVBR", "PVCL", "PVCR"]

import importlib
import c302
import c302_gen_reader
sys.modules["c302.c302_gen_reader"] = c302_gen_reader
from pyneuroml import pynml


def names_of():
    proc = np.load(PROC, allow_pickle=True); g = np.load(GEN, allow_pickle=True)
    na = [str(x) for x in proc["node_names"]]
    return [na[i] for i in g["node_idx"]]


def find_trace(data, name):
    for k in data:
        if k == "t": continue
        if name in k.replace("[", "/").replace("]", "/").split("/"):
            return np.array(data[k])
    return None


def run_mode(mode, names):
    os.environ["C302_CONN_MODE"] = mode
    target = os.path.join(OUTDIR, f"c302_inst_{mode}"); os.makedirs(target, exist_ok=True)
    ref = f"c302_inst_{mode}"
    params = getattr(importlib.import_module(f"c302.parameters_{PSET}"), "ParameterisedModel")()
    ant = [n for n in ANT if n in names]
    plot = [n for n in BACK + FWD + ANT if n in names]
    ov = {"unphysiological_offset_current": "12pA",
          "neuron_to_neuron_chem_exc_syn_gbase": "0.5nS",
          "neuron_to_neuron_chem_inh_syn_gbase": "0.5nS"}
    c302.generate(ref, params, data_reader="c302_gen_reader", cells=None,
                  cells_to_plot=plot, cells_to_stimulate=ant, muscles_to_include=[],
                  duration=DUR, dt=0.05, target_directory=target, param_overrides=ov, verbose=False)
    data = pynml.run_lems_with_jneuroml(f"LEMS_{ref}.xml", exec_in_dir=target,
                                        nogui=True, load_saved_data=True, plot=False, verbose=False)
    def act(grp):
        vs = [np.mean(find_trace(data, n)) for n in grp if n in names and find_trace(data, n) is not None]
        return float(np.mean(vs)) if vs else float("nan")
    b, f = act(BACK), act(FWD)
    return b, f


def main():
    names = names_of()
    print("[#4 c302 본능] 앞터치 자극 → 명령뉴런 평균 탈분극", flush=True)
    res = {}
    for mode in ["real", "gen"]:
        b, f = run_mode(mode, names)
        res[mode] = (b, f, b - f)
        print(f"  {mode}: 후진명령 {b:.4f}, 전진명령 {f:.4f}, 차(후진-전진) {b-f:+.4f}", flush=True)
    print(f"\n실제 차 {res['real'][2]:+.4f} / 생성 차 {res['gen'][2]:+.4f}")
    print("(>0 = 후진/회피 우세 = escape 반사. c302 생물물리로도 보이나)")


if __name__ == "__main__":
    main()
