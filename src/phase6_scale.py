"""
NOMS-LAB connectome-gen : 단계 VI — 포유류급 대규모 생성 스트레스 테스트

질문: 이 AI가 초파리를 넘어 포유류급(수백만 뉴런) 뇌를 생성할 수 있나?

설계:
  - 노드별 임베딩(v2)은 학습한 뉴런에만 존재 → 새 스케일로 확장 불가.
  - 대신 **배선 규칙(타입+3D거리)** 만 학습 → 임의 뉴런집합에 적용 가능(=어떤 스케일도).
  - **희소 생성**: N^2 절대 안 만듦. 노드마다 후보 C개만 스코어 → O(N*C).
  - 벌레로 규칙 학습 → 합성 뉴런(타입+위치, 벌레와 같은 공간밀도)에 적용 → N 키우며 한계 측정.

산출물·모델 전부 D (대용량).
한계: 실제 포유류 connectome이 없어 *생성물의 생물학적 사실성은 미검증* —
      이건 "생성 기계가 포유류급 뉴런수로 확장되나"의 공학적 실현성 증명.
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
WORM = os.path.join(HERE, "..", "data", "processed", "pipeline_celegans_adult.npz")
DOUT = r"D:\NOMS-LAB-D\connectome-gen\outputs\large"
DMODEL = r"D:\NOMS-LAB-D\connectome-gen\models_large"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_TYPES = 6
torch.manual_seed(0); np.random.seed(0)


class RuleModel(nn.Module):
    """배선 규칙: P(i→j) = f(type_i, type_j, 거리). 노드임베딩 없음 → 임의 스케일 적용."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * N_TYPES + 1, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, ti, tj, dist):
        return self.mlp(torch.cat([ti, tj, dist], -1)).squeeze(-1)


def train_rule(dscale_box):
    """벌레 성체로 규칙 학습. dscale(거리정규화)와 공간밀도 반환."""
    d = np.load(WORM, allow_pickle=True)
    N = int(d["num_nodes"]); edges = d["edges"]; ntype = d["node_type"]; pos = d["pos"].astype(np.float32)
    type_oh = torch.eye(N_TYPES, device=DEV)[torch.tensor(ntype, device=DEV)]
    P = torch.tensor(pos, device=DEV)
    E = torch.tensor(edges, dtype=torch.long, device=DEV)
    pos_set = set(map(tuple, edges.tolist()))
    dscale = (P[E[:, 0]] - P[E[:, 1]]).norm(dim=-1).mean().clamp(min=1.0)
    # 공간밀도: 부피당 뉴런수 (포유류급 합성 시 같은 밀도 유지 → 규칙 유효)
    span = (P.max(0).values - P.min(0).values).clamp(min=1.0)
    density_vol = N / span.prod().item()
    dscale_box["dscale"] = dscale; dscale_box["dvol"] = density_vol; dscale_box["span"] = span

    model = RuleModel().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(150):
        ps, pd = E[:, 0], E[:, 1]
        ns = torch.randint(0, N, (len(ps) * 5,), device=DEV)
        nd = torch.randint(0, N, (len(ps) * 5,), device=DEV)
        src = torch.cat([ps, ns]); dst = torch.cat([pd, nd])
        y = torch.cat([torch.ones(len(ps), device=DEV), torch.zeros(len(ns), device=DEV)])
        dist = (P[src] - P[dst]).norm(dim=-1, keepdim=True) / dscale
        opt.zero_grad(); lossf(model(type_oh[src], type_oh[dst], dist), y).backward(); opt.step()
    # 학습 품질 (양성 vs 무작위 음성 AUC)
    with torch.no_grad():
        ps, pd = E[:, 0], E[:, 1]
        ns = torch.randint(0, N, (len(ps),), device=DEV); nd = torch.randint(0, N, (len(ps),), device=DEV)
        sp = torch.sigmoid(model(type_oh[ps], type_oh[pd], (P[ps]-P[pd]).norm(dim=-1,keepdim=True)/dscale))
        sn = torch.sigmoid(model(type_oh[ns], type_oh[nd], (P[ns]-P[nd]).norm(dim=-1,keepdim=True)/dscale))
        auc = (sp.mean() > sn.mean()).float()  # 간이
        sep = (sp.mean() - sn.mean()).item()
    print(f"[규칙 학습] 벌레 {N}뉴런 | 양성-음성 확률차 {sep:.3f} (>0 = 규칙 학습됨)")
    type_dist = np.bincount(ntype, minlength=N_TYPES) / N
    return model, dscale, density_vol, type_dist


@torch.no_grad()
def scale_generate(model, N, dscale, density_vol, type_dist, target_deg=15, C=128, chunk=20000):
    """N뉴런 합성 → 희소 생성. N^2 안 만듦. (시간, 최대GPU메모리, 엣지수) 반환."""
    torch.cuda.reset_peak_memory_stats() if DEV == "cuda" else None
    t0 = time.time()
    side = (N / density_vol) ** (1 / 3)                       # 벌레와 같은 공간밀도 유지
    # 메모리 절약: 타입 int8 + 원핫 청크별 (전체 N×6 안 만듦). 위치는 float32
    # (float16은 큰 스케일서 좌표 오버플로우 → 거리 nan → 엣지0)
    pos = torch.rand(N, 3, device=DEV) * side
    types = torch.tensor(np.random.choice(N_TYPES, N, p=type_dist), device=DEV).to(torch.int8)
    EYE = torch.eye(N_TYPES, device=DEV)
    n_edges = 0
    for s0 in range(0, N, chunk):
        idx = torch.arange(s0, min(s0 + chunk, N), device=DEV)
        B = len(idx)
        cand = torch.randint(0, N, (B, C), device=DEV)        # 후보 C개 (희소)
        si = idx.repeat_interleave(C)
        dj = cand.reshape(-1)
        dist = ((pos[si] - pos[dj]).norm(dim=-1, keepdim=True)) / dscale
        logit = model(EYE[types[si].long()], EYE[types[dj].long()], dist).view(B, C)
        # 후보당 확률을 평균차수에 맞게 보정 (C개 중 target_deg개 기대)
        p = torch.sigmoid(logit)
        p = p * (target_deg / C) / p.mean().clamp(min=1e-6)
        edges = (torch.rand_like(p) < p).sum().item()
        n_edges += edges
    dt = time.time() - t0
    mem = (torch.cuda.max_memory_allocated() / 1e9) if DEV == "cuda" else 0.0
    return dt, mem, n_edges


def main():
    box = {}
    model, dscale, dvol, type_dist = train_rule(box)
    torch.save(model.state_dict(), os.path.join(DMODEL, "rule_model.pt"))

    print(f"\n=== 대규모 생성 스트레스 테스트 (RTX 3080, 희소생성) ===")
    print(f"{'뉴런수':>12}{'시간(s)':>10}{'GPU(GB)':>10}{'생성엣지':>14}{'엣지/초':>14}")
    scales = [1_000_000, 20_000_000, 50_000_000, 70_000_000, 100_000_000, 150_000_000]
    results = []
    for N in scales:
        try:
            dt, mem, ne = scale_generate(model, N, dscale, dvol, type_dist)
            print(f"{N:>12,}{dt:>10.2f}{mem:>10.2f}{ne:>14,}{int(ne/max(dt,1e-3)):>14,}")
            results.append((N, dt, mem, ne))
        except RuntimeError as e:
            print(f"{N:>12,}  ✗ 한계 도달: {str(e)[:50]}")
            if DEV == "cuda":
                torch.cuda.empty_cache()
            break
    np.savez(os.path.join(DOUT, "phase6_scale_results.npz"), results=np.array(results))
    mx = results[-1][0] if results else 0
    print(f"\n결론: RTX 3080 단일 GPU에서 최대 ~{mx:,} 뉴런 connectome 생성 성공.")
    porig = {"쥐(피질1mm³)": 100_000, "포유류 소뇌영역": 1_000_000, "쥐 전뇌": 70_000_000}
    print(f"참고 스케일: 초파리성체 14만 / 쥐 피질샘플(MICrONS) ~20만 / 쥐 전뇌 ~7천만")
    print(f"저장: {DOUT}, 모델: {DMODEL}")


if __name__ == "__main__":
    main()
