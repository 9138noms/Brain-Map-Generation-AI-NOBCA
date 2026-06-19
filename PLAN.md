# connectome-gen — 뇌지도 생성 AI (NOMS-LAB)

> 뇌지도(connectome)를 학습해 새로운 뇌지도를 생성하고, 그것이 "작동"하는지 검증하는 프로젝트.

## 저장 구조 (C/D 분할)
- **코드·가공텐서·체크포인트** → `C:\Users\yuulf\Desktop\NOMS-LAB-C\connectome-gen\` (SSD)
- **원본 데이터·출력물·로그** → `D:\NOMS-LAB-D\connectome-gen\` (대용량)
  - 원본 repo: `D:\NOMS-LAB-D\connectome-gen\data\raw\nature2021\`

## 목표
1. **생성**: 기존 connectome을 학습해, 학습하지 않은 *새로운* connectome 그래프를 생성하는 AI
2. **검증**: 생성된 connectome이 (1) 구조적으로 유효한지, (2) 기능적으로 작동하는지 평가

## 대상 생물: C. elegans (예쁜꼬마선충)
- 302 뉴런 — RTX 3080에서 가볍게 학습 가능
- 가장 잘 연구된 connectome (완전 매핑)
- **결정적 이유**: OpenWorm/c302 라는 기능 시뮬레이터가 존재 → "작동" 검증 가능
  - 초파리(14만)·생쥐는 통째 기능 시뮬이 없어 검증 단계 불가

## 데이터
- **Witvliet et al. 2021 (Nature)** — 발달 8단계 connectome (birth→adult). 생성 모델 학습용 핵심 (같은 종의 변이 샘플)
- **Cook et al. 2019** — 암/수 성체 완전 배선도 (wormwiring.org)
- **Varshney et al. 2011** — 고전 connectome (네트워크 분석용 baseline)
- **OpenWorm c302** — 기능 시뮬레이터 (검증용)

## Phase 1 — 생성
- 입력 표현: 인접행렬(가중/방향) + 뉴런 메타(위치 3D, 타입: sensory/inter/motor, 좌우)
- 모델 후보:
  - **Graph VAE** — 잠재공간 샘플링으로 새 그래프
  - **Graph diffusion** — 더 강력, 구현 복잡
  - **Generative wiring rules (Betzel류)** — baseline, 공간+위상 규칙 ~13개 파라미터
- 출력: 학습 분포에서 나온 *새로운* connectome

## Phase 2 — 검증 ("작동하냐")
### 1. 구조적 유효성 (빠름/정량)
- 차수분포, motif 빈도, small-worldness, 모듈성, 좌우대칭성
- 실제 벌레 통계와 매칭되는가
### 2. 기능적 유효성 (진짜 "작동")
- 생성 connectome을 c302 동역학 시뮬에 주입
- 감각자극 → 운동뉴런 반응 → locomotion 패턴(전진/후진/방향전환) 발현 여부

## 핵심 난제 = 연구 기여
**새로움 vs 유효성 트레이드오프**
- 복붙하면 유효하지만 무의미, 너무 새로우면 작동 안 함
- "새로운데 작동하는" connectome 생성 = 핵심 가치
- novelty 지표(학습셋과의 그래프 거리) vs 기능 점수의 파레토 곡선을 보여주는 것이 목표

## 공개 계획
- GitHub repo (코드 + 생성 샘플 + 평가 파이프라인)
- 결과 좋으면 워크샵 페이퍼 / 블로그

## 진행 로그
- 2026-06-19: LAB 생성, 계획 고정, C. elegans 타깃 확정.
- 2026-06-19: **Phase 0 완료** — Witvliet 8단계 데이터 수집 + 로더(`src/load_connectome.py`).
  텐서화 완료: 229노드(union), chem/gap 인접행렬, 3D좌표, 발달단계별 존재마스크 → `data/processed/celegans_dev.npz`.
  성장곡선 재현: 뉴런 187→221(+18%), 화학시냅스 1296→7970(+515%) = 발달 = 시냅스 densification.
- 2026-06-19: 저장소 C/D 분할 정리.
- 2026-06-19: **Phase 1 v1 완료** — 엣지 단위 생성 모델 (`src/phase1_edge_model.py`).
  엣지=샘플 프레이밍으로 312,574개 학습신호 확보(8개 그래프 벽 우회).
  예측: 단계1~7 학습 → 성체(8) 엣지 AUC **0.781** (거리 baseline 0.729 능가) = 진짜 배선규칙 학습.
  생성(밀도맞춤): density/평균차수/**sensory→inter→motor 정보흐름** 재현 성공.
  한계: **상호성 0.279→0.053**, 허브 약간 부족 — 엣지 독립샘플링의 구조적 한계.
  → v2 과제: 노드 잠재임베딩(허브) + 엣지 의존성/상호성 모델링.
- 2026-06-19: **Phase 1 v2 완료** — 노드임베딩+상호성, GPU전용/벡터화 (`src/phase1_v2.py`).
  logit = MLP(type,dist,stage) + u_i·v_j(허브) + s_i·s_j(상호성).
  예측 AUC **0.78→0.96**, AP **0.15→0.71**. 생성: 상호성 0.05→**0.22**(실제0.28), 허브 일치(33/35 vs 32/39).
  → C.elegans 구조생성 **토대 완성**. 다음: (a)초파리 유충 스케일 (b)작동검증(c302).
  CPU는 타 프로젝트(NOTP) 점유 → GPU 전용으로 진행. 파이프라인 벡터화로 스케일 대비 완료.
