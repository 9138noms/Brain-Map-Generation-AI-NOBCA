"""
NOMS-LAB connectome-gen : 단계 XX — 대용량 뇌 생성 (~300GB) → D 저장

쥐 피질 규칙 학습 → ~3.7억 뉴런(피질 평균차수 80) 스트리밍 생성 → D에 바이너리 저장.
N^2 안 만들고, 청크마다 엣지를 디스크에 직접 기록(메모리 안전). 목표 용량 도달시 정지.
엣지 포맷: (src uint32, dst uint32, weight uint16) = 10 bytes.

사용: py -3.12 src/phase20_bigbrain_300gb.py [target_GB] [N]
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
MOUSE = os.path.join(HERE, "..", "data", "processed", "pipeline_microns_mouse.npz")
DOUT = r"D:\NOMS-LAB-D\connectome-gen\outputs\large"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

TARGET_GB = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000_000
BYTES_PER_EDGE = 10
EDGE_DT = np.dtype([("src", np.uint32), ("dst", np.uint32), ("w", np.uint16)])
CHUNK = 32_000           # source 노드 청크 (CHUNK*C 쌍이 MLP에 한번에 → 메모리 주의)
C = 128                  # 후보 수
N_TYPES = 12
OUTFILE = os.path.join(DOUT, f"bigbrain_{int(TARGET_GB)}gb.bin")


class Rule(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * T + 1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, ti, tj, dist):
        return self.mlp(torch.cat([ti, tj, dist], -1)).squeeze(-1)


def train_rule():
    d = np.load(MOUSE, allow_pickle=True)
    Nm = int(d["num_nodes"]); E = d["edges"]; nt = d["node_type"]; pos = d["pos"]
    T = int(nt.max()) + 1
    P = torch.tensor(pos, dtype=torch.float32, device=DEV)
    toh = torch.eye(T, device=DEV)[torch.tensor(nt, device=DEV)]
    Et = torch.tensor(E, dtype=torch.long, device=DEV)
    dscale = (P[Et[:, 0]] - P[Et[:, 1]]).norm(dim=-1).mean().clamp(min=1.0)
    model = Rule(T).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    lf = nn.BCEWithLogitsLoss()
    for ep in range(100):
        ps, pd = Et[:, 0], Et[:, 1]
        ns = torch.randint(0, Nm, (len(ps) * 5,), device=DEV); nd = torch.randint(0, Nm, (len(ps) * 5,), device=DEV)
        src = torch.cat([ps, ns]); dst = torch.cat([pd, nd])
        dist = (P[src] - P[dst]).norm(dim=-1, keepdim=True) / dscale
        y = torch.cat([torch.ones(len(ps), device=DEV), torch.zeros(len(ns), device=DEV)])
        opt.zero_grad(); lf(model(toh[src], toh[dst], dist), y).backward(); opt.step()
    span = (P.max(0).values - P.min(0).values).clamp(min=1.0)
    return model, T, float(dscale), Nm / span.prod().item(), np.bincount(nt, minlength=T) / Nm, len(E) / Nm


def main():
    model, T, dscale, dvol, tdist, mean_deg = train_rule()
    target_bytes = TARGET_GB * 1e9
    print(f"목표 {TARGET_GB}GB | 뉴런 {N:,} | 평균차수~{mean_deg:.0f} | 파일 {OUTFILE}", flush=True)

    side = ((N / dvol) ** (1 / 3)) / dscale   # 연결거리 단위로 정규화 (fp16 오버플로 방지)
    print("뉴런 배치 중...", flush=True)
    pos = (torch.rand(N, 3, device=DEV) * side).half()
    types = torch.tensor(np.random.choice(T, N, p=tdist), device=DEV, dtype=torch.int8)
    EYE = torch.eye(T, device=DEV)
    cs = ((300 / dvol) ** (1 / 3)) / dscale; G = int(side / cs) + 2
    cell = (pos.float() / cs).floor().to(torch.int32).clamp(0, G - 1)
    cid = cell[:, 0] * (G * G) + cell[:, 1] * G + cell[:, 2]   # int32, 셀수<2^31
    order = torch.argsort(cid).to(torch.int32)
    target_deg = target_bytes / (N * BYTES_PER_EDGE)           # N노드로 300GB 채우게 자동 차수
    uniq, inv, cnt = torch.unique(cid[order.long()], return_inverse=True, return_counts=True)
    inv = inv.to(torch.int32)   # 셀수 < 2^31 → int32로 메모리 절감 (3.7억 노드 OOM 방지)
    starts = (torch.cumsum(cnt, 0) - cnt).to(torch.int32)
    del cid, cell
    torch.cuda.empty_cache() if DEV == "cuda" else None

    f = open(OUTFILE, "wb")
    written = 0; n_edges = 0; t0 = time.time(); last = t0
    for p0 in range(0, N, CHUNK):
        ps = torch.arange(p0, min(p0 + CHUNK, N), device=DEV)
        ic = inv[ps].long(); st = starts[ic].long(); ln = cnt[ic].clamp(min=1)
        off = (torch.rand(len(ps), C, device=DEV) * ln.unsqueeze(1)).long()
        cand = order[(st.unsqueeze(1) + off).reshape(-1)].long()
        si = order[ps].long().repeat_interleave(C)
        with torch.no_grad():
            dist = (pos[si].float() - pos[cand].float()).norm(dim=-1, keepdim=True)   # 이미 정규화됨
            p = torch.sigmoid(model(EYE[types[si].long()], EYE[types[cand].long()], dist)).view(len(ps), C)
        p = p * (target_deg / C) / p.mean().clamp(min=1e-6)
        fire = torch.rand_like(p) < p
        rows, cols = fire.nonzero(as_tuple=True)
        src = order[ps][rows].cpu().numpy().astype(np.uint32)
        dst = cand.view(len(ps), C)[rows, cols].cpu().numpy().astype(np.uint32)
        w = (1 + np.random.geometric(0.3, len(src)).clip(1, 60)).astype(np.uint16)  # 시냅스수 모사
        arr = np.empty(len(src), EDGE_DT); arr["src"] = src; arr["dst"] = dst; arr["w"] = w
        f.write(arr.tobytes())
        written += arr.nbytes; n_edges += len(src)
        if time.time() - last > 30:
            gb = written / 1e9; rate = gb / (time.time() - t0)
            eta = (target_bytes / 1e9 - gb) / max(rate, 1e-9) / 60
            print(f"  {gb:.1f}GB / {TARGET_GB}GB | {n_edges:,}엣지 | {rate*60:.1f}GB/분 | ETA {eta:.0f}분", flush=True)
            last = time.time()
        if written >= target_bytes:
            break
    f.close()
    dt = time.time() - t0
    np.savez(os.path.join(DOUT, f"bigbrain_{int(TARGET_GB)}gb_meta.npz"),
             N=N, edges=n_edges, bytes=written, type_dist=tdist, T=T, seconds=dt)
    print(f"\n완료: {written/1e9:.1f}GB, {n_edges:,}엣지, {N:,}뉴런, {dt/60:.1f}분", flush=True)
    print(f"저장: {OUTFILE}", flush=True)


if __name__ == "__main__":
    main()
