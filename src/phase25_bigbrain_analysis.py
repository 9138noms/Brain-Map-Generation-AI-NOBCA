"""
NOMS-LAB connectome-gen : 단계 XXV — 245GB 큰 뇌 연구 (스트리밍 분석)

질문: 3억 뉴런 규칙생성 뇌에 *창발적 뇌-유사 구조*가 나타나나?
245GB 엣지파일을 청크 스트리밍(메모리 안전)으로 1회 통과:
  - 전체 차수분포(in/out) → heavy-tail/허브
  - 자기연결 비율, 시냅스수 분포
  - 무작위 샘플 노드들의 유도 부분그래프 → 군집계수·상호성·밀도
실제 쥐 피질(2220)과 비교 → 135,000배 스케일서 구조 보존되나.
"""
import os, sys, time
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = r"D:\NOMS-LAB-D\connectome-gen\outputs\large\bigbrain_300gb_layered.bin"
MOUSE = os.path.join(HERE, "..", "data", "processed", "pipeline_microns_mouse.npz")
OUT = r"D:\NOMS-LAB-D\connectome-gen\outputs\large\phase25_analysis.npz"
N = 300_000_000
EDGE_DT = np.dtype([("src", np.uint32), ("dst", np.uint32), ("w", np.uint16)])
CHUNK = 50_000_000          # 엣지/청크 (500MB)
np.random.seed(0)


def main():
    out_deg = np.zeros(N, dtype=np.int32)
    in_deg = np.zeros(N, dtype=np.int32)
    # 부분그래프용 샘플 노드 5000개 멤버십
    S = np.random.choice(N, 5000, replace=False)
    member = np.zeros(N, dtype=bool); member[S] = True
    sg_src, sg_dst = [], []
    wsum = 0.0; wmax = 0; selfloop = 0; total = 0
    t0 = time.time()
    fsize = os.path.getsize(BIN)
    print(f"파일 {fsize/1e9:.1f}GB 스트리밍 분석 시작...", flush=True)
    with open(BIN, "rb") as f:
        ci = 0
        while True:
            buf = f.read(CHUNK * EDGE_DT.itemsize)
            if not buf:
                break
            e = np.frombuffer(buf, dtype=EDGE_DT)
            src = e["src"].astype(np.int64); dst = e["dst"].astype(np.int64); w = e["w"]
            out_deg += np.bincount(src, minlength=N).astype(np.int32)
            in_deg += np.bincount(dst, minlength=N).astype(np.int32)
            selfloop += int((src == dst).sum()); total += len(e)
            wsum += float(w.sum()); wmax = max(wmax, int(w.max()))
            mask = member[src] & member[dst]
            if mask.any():
                sg_src.append(src[mask]); sg_dst.append(dst[mask])
            ci += 1
            if ci % 5 == 0:
                print(f"  {total/1e9:.1f}B엣지 처리, {time.time()-t0:.0f}s", flush=True)

    # 차수분포
    print(f"\n=== 차수분포 (3억 뉴런) ===", flush=True)
    print(f"  평균 out차수 {out_deg.mean():.1f} | 최대 {out_deg.max()} | std {out_deg.std():.1f}")
    print(f"  평균 in차수  {in_deg.mean():.1f} | 최대 {in_deg.max()} | std {in_deg.std():.1f}")
    # heavy-tail 지표: 상위1% 노드가 가진 엣지 비율
    thr = np.percentile(out_deg, 99)
    top_frac = out_deg[out_deg >= thr].sum() / out_deg.sum()
    print(f"  상위1% 뉴런이 가진 출력엣지 비율: {top_frac*100:.1f}% (높을수록 허브집중)")
    print(f"  고립뉴런(차수0): {(((out_deg+in_deg)==0).sum())/N*100:.2f}%")
    print(f"  자기연결: {selfloop/total*100:.3f}% | 시냅스수 평균 {wsum/total:.2f} 최대 {wmax}")

    # 부분그래프
    ss = np.concatenate(sg_src) if sg_src else np.array([], np.int64)
    sd = np.concatenate(sg_dst) if sg_dst else np.array([], np.int64)
    idx = {n: i for i, n in enumerate(S)}
    k = len(S); A = np.zeros((k, k), np.float32)
    for a, b in zip(ss, sd):
        if a in idx and b in idx and a != b:
            A[idx[a], idx[b]] = 1
    ne = A.sum()
    recip = (A * A.T).sum() / max(ne, 1)
    U = ((A + A.T) > 0).astype(float)
    cl = []
    for i in range(k):
        nb = np.where(U[i] > 0)[0]
        if len(nb) >= 2:
            cl.append(U[np.ix_(nb, nb)].sum() / (len(nb) * (len(nb) - 1)))
    print(f"\n=== 샘플 부분그래프 ({k}뉴런) ===")
    print(f"  내부 엣지 {int(ne)} | 상호성 {recip:.3f} | 군집계수 {np.mean(cl) if cl else 0:.3f}")

    # 실제 쥐 비교
    m = np.load(MOUSE, allow_pickle=True)
    Nm = int(m["num_nodes"]); E = m["edges"]
    od = np.bincount(E[:, 0], minlength=Nm)
    print(f"\n=== 실제 쥐 피질(2220) 대조 ===")
    print(f"  평균차수 {len(E)/Nm:.1f} | 상위1% 비율 {od[od>=np.percentile(od,99)].sum()/od.sum()*100:.1f}%")

    np.savez(OUT, out_deg_hist=np.bincount(out_deg.clip(0, 500)),
             selfloop_frac=selfloop/total, mean_deg=out_deg.mean(), top1pct=top_frac)
    print(f"\n저장: {OUT} | 총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
