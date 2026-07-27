# 워커 + MedGemma 인터페이스 (담당자 E: 박검모)

팀원 연동용 문서. "내가 무엇을 받고, 무엇을 주는가"를 정의한다.
발표 15·16·17번 슬라이드 기준.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `til_conformal.py` | sTIL + Conformal Prediction 계산 (GPU 불필요, 순수 통계) |
| `medgemma_report.py` | MedGemma 호출 + 5절 규격 JSON 조립 + 파싱 재시도/폴백 |
| `worker.py` | 진입점 — 위 둘을 엮어 케이스 1건 처리 |
| `test_til_conformal.py` | 계산 로직 검증 (9개 테스트, 전부 통과) |

## 🔴 최일운 (Mosec) 에게 요청 — 내가 받는 입력

`worker.process_case()`가 기대하는 것:

| 입력 | 타입 | 설명 |
|------|------|------|
| `mc_stack` | numpy `(N, H, W)` uint8 | MC-Dropout 세그마스크 **N개 스택**. ⚠️ 평균/argmax 하지 말고 **N개 원본 예측을 그대로** 주세요 (불확실도 계산에 필요) |
| `seg_image` | PIL.Image | MedGemma 입력용 세그멘테이션 결과 이미지 |
| `model_name` | str | 현재 보정 완료: `attention_unet_v1` **1종** (아래 🔴 참고) |

**🔴 모델 선택 기능 관련 (2026-07-26) — 정다희·윤수용 함께 확인 필요**
추후 병리의가 U-Net / Attention U-Net 등을 골라 비교할 수 있게 선택 UI를 구현할
예정인데, **conformal 보정상수 q는 모델마다 따로 산출해야 하는 값**입니다.
보정 안 된 모델을 선택지에 노출하면 그 모델의 예측구간이 Attention U-Net의
q(19.63)로 계산되어 **커버리지 보장이 깨진 채 화면에 나갑니다.**
→ 워커는 `get_conformal_q()`에서 미보정 모델을 **에러로 차단**하도록 구현했습니다
  (조용한 fallback 없음 — 잘못된 판정을 내보내느니 실패시킴).
→ **프론트에서도 보정 완료된 모델만 선택 가능하게 해주세요.** 미보정 모델을
  드롭다운에 넣으면 워커가 실패하고 Job이 `실패` 상태로 끝납니다.
→ 새 모델을 선택지에 추가하려면: calibration으로 q 산출 → `worker.CONFORMAL_Q`에 등록
  → 그 다음 프론트 드롭다운에 노출, 순서를 지켜야 합니다.

**클래스 인덱스 컨벤션 (반드시 일치):**
`0=tumor, 1=stroma, 2=immune, 3=necrosis, 4=other, 255=ignore`

**✅ 2026-07-24 실제 데이터로 검증 완료** — 최일운의 실제 U-Net 출력 npy
(`unet_TCGA-A2-A3XX-DX1_..._MPP-0.2500.npy`)로 확인:
- `shape (20, 2529, 3028)`, `dtype uint8`, `unique [0 1 2 3 4]` — 위 기대 스펙과 완전히 일치
- `compute_stil_and_conformal()`에 그대로 넣어서 `stil_value=4.75, decision="LOW"` 정상 산출
- 포맷 불일치 리스크는 해소됨. 다만 **컷오프(50%) 근처 케이스에서 abstain=True가 뜨는지는 아직 미검증** (이번 샘플은 4.75%라 쉬운 케이스였음) — 근처 값 나는 다른 슬라이드로 한 번 더 확인 필요

**전달 방식 미정:** HTTP 직접 호출 vs GCS 파일 경로. → 윤수용과 함께 확정.

## 🔴 정다희 (React) 에게 제공 — 내가 주는 출력

`Job.medgemma_report`에 저장되는 JSON (그대로 화면에 꽂으면 됨):

```json
{
  "case_id": "5482ca2e",
  "model_used": "unet_v1",
  "stil_ratio": 47.0,
  "uncertainty": {
    "cp_value": 0.86,
    "lower": 38.0,
    "upper": 56.0,
    "abstain": true
  },
  "class_composition": {
    "tumor": 42.1, "stroma": 34.2, "immune": 12.3, "necrosis": 8.1, "other": 3.3
  },
  "segmentation_summary": "조직 이미지에서 유방 조직과 암세포가 혼합되어...",
  "findings": "sTIL 47%로 경계 부위에서 국소적 침윤 양상이...",
  "disclaimer": "본 결과는 AI 보조 도구의 출력이며 의료진의 최종 판단이 필요합니다."
}
```

**`class_composition` (2026-07-27 추가) — 화면 4 도넛차트용:**
- 키 5개 고정(`tumor`/`stroma`/`immune`/`necrosis`/`other`), 값은 % (소수 1자리)
- 255(ignore) 픽셀은 분모에서 제외 → **5개 값의 합은 100%**
  ⚠️ 반올림 때문에 99.9~100.1로 나올 수 있습니다. 도넛 각도 계산 시 합으로
  정규화하거나, 라벨은 그대로 쓰고 각도만 비율로 계산하세요.
- MC-Dropout 20회의 **다수결(consensus) 마스크** 기준 — 화면에 보이는 세그멘테이션
  마스크의 구성비와 일치합니다.
- ⚠️ sTIL은 회차별 값의 **중앙값**이라 계산 경로가 다릅니다. 따라서
  `immune/(immune+stroma)`를 이 구성비로 다시 계산하면 `stil_ratio`와 소수점 단위로
  어긋날 수 있습니다 — 버그가 아니라 두 지표의 정의 차이입니다. **화면에서 둘을
  나란히 놓고 "계산이 맞는지" 검산하도록 유도하지 마세요.**
- 조직이 전혀 없으면 `null`입니다 (도넛 대신 "구성비 산출 불가" 표시 필요)

**화면 필드 매핑:**
| React 화면 요소 | JSON 필드 |
|----------------|-----------|
| sTIL 큰 숫자 | `stil_ratio` |
| CP 값 (0.86) | `uncertainty.cp_value` |
| 예측구간 막대 | `uncertainty.lower` ~ `upper` |
| 기권 뱃지 | `uncertainty.abstain` (true면 빨간 "기권") |
| 5-class 도넛차트 | `class_composition` (5개 키, 합 100%) |
| VLM 판독결과지 | `findings` |
| 요약 | `segmentation_summary` |
| 하단 주의문구 | `disclaimer` |

**🔴 협의 필요 (2026-07-26 갱신) — `cp_value`는 화면에 쓰지 마세요:**
Conformal이 **비정규화**로 확정되면서(최일운 최종 보고 5절) 모든 케이스의 구간 폭이
`2q = 39.26`으로 동일해졌습니다. 그 결과 `cp_value`는 **모든 케이스에서 상수 0.203**이고,
sTIL이 0%/100% 경계에 가까워 구간이 잘릴 때만 값이 달라집니다:

| sTIL | cp_value | 구간 폭 |
|------|----------|---------|
| 30 / 40 / 50 / 60 / 70% | **0.203 (전부 동일)** | 39.26 |
| 10% | 0.2523 | 29.63 (경계 clipping) |
| 2% | 0.3162 | 21.63 (경계 clipping) |

즉 지금의 `cp_value`는 신뢰도 정보를 담지 못할 뿐 아니라, **sTIL이 0%에 가까운 케이스가
더 확신 있어 보이는 착시**를 만듭니다. 발표 24번 슬라이드의 "CP 값 0.86" 형태 표시는
비정규화 전환 이후 성립하지 않습니다.
→ **대안:** 화면에는 `lower~upper` 구간과 `abstain` 배지만 쓰는 것을 권장.
   (필드 자체는 스키마 호환을 위해 유지하되 표시하지 않음)

## 🔴 윤수용 (인프라) 에게 요청

1. ✅ **확정 (2026-07-24)** — **Job 테이블 쓰기 방법**: **Option B** (워커는 DB 모름, Django API에 HTTP PATCH).
   `worker.update_job(job_id, **fields)` 구현 완료 — `requests.patch(f"{DJANGO_API_BASE}/jobs/{job_id}/", json=fields)`.
   `DJANGO_API_BASE`는 배포 시 실제 Django API 주소로 환경변수 주입 필요 (현재 플레이스홀더).
2. ✅ **확정 (2026-07-24)** — **mc_stack/seg_image 전달 방식**: **Option B (GCS 경로)**.
   mc_stack이 실측 약 146MB라 Cloud Tasks 페이로드 100KB 제한상 데이터 직송은 불가능 — 페이로드엔
   `mc_stack_path`/`gcs_result_path`로 `gs://...` 문자열만 오고, 워커가 직접 다운로드.
   `worker.load_mc_stack()` / `worker.load_seg_image()` 구현 완료, `run_job()`에 연결 완료.
   **남은 것:** 정확한 페이로드 필드명(현재 `job_id`/`model_name`/`mc_stack_path`/`gcs_result_path`로 가정 중)과
   버킷 경로 규칙(`gs://버킷/jobs/{job_id}/...` 등)은 여전히 협의 필요.
3. **MedGemma Docker**: 가중치를 빌드타임에 포함하는 구조(발표 16번) → Dockerfile 같이 검토
5. **Cloud Run Job 메모리**: mc_stack `(20, 2529, 3028)` 1건 처리 시 워커 측 실측
   피크 RSS **약 0.4GB** (입력 0.15GB + 구성비 계산 0.22GB). 여기에 MedGemma 모델이
   따로 올라가므로, Job 메모리는 그 합보다 여유 있게 잡아주세요.
   (BF16 로드 시 가중치만 ~8GB — int8 양자화 시 ~4GB, `load_pipeline()` 주석 참고)
4. ✅ **확정 (2026-07-24)** — **상태 전이**: `status="sTIL 완료"` → MedGemma 후 `status="완료"` 규약대로
   `run_job()`에 두 번의 `update_job()` 호출로 구현 완료.

## ⚠️ 아직 placeholder (실제 값으로 교체 필요)

- ✅ **해소 (2026-07-26)** — `CONFORMAL_Q`가 실제 calibration 값으로 교체됨.
  `attention_unet_v1 = 19.63` (CV+, 비정규화, α=0.10, MC Dropout 20회 기준).
  Held-out 6개 기관 45 ROI에서 커버리지 95.6%로 목표 90% 충족 확인.
  ⚠️ **q는 sigma 배수가 아니라 구간 절반폭(%p) 자체**이며, MC 20회 조건에 묶여 있음 —
  MC 횟수를 바꾸면 q를 다시 보정해야 함.
- `medgemma_report.load_pipeline()` — 현재 BF16 그대로 로드. VRAM이 부족한 배포 환경(예: RTX 3070 8GB)에서는
  함수 안 bitsandbytes int8 양자화 블록의 주석만 해제하면 바로 적용됨 (`requirements.txt`의 `bitsandbytes`도 주석 해제).
  **켠 뒤에는 sTIL 할루시네이션 재검증 필수** — 양자화로 정밀도가 바뀌면 생성 결과가 미세하게 달라질 수 있음.
- `worker.py`의 `CONFORMAL_Q` — 모델별 conformal 보정상수 q가 지금 더미값(1.64).
  실제로는 calibration 세트로 모델마다 계산해야 함. (전처리 때 만든 CV+ fold 로직 재사용)
- `findings` 프롬프트 품질 — 아직 입력값 재진술 경향. 튜닝 예정.
- **🔴 할루시네이션 발견 (2026-07-24)**: 실제 npy로 테스트 중 MedGemma가
  `segmentation_summary`에서 "sTIL(시스플라스크 림프관 세포)"라는 존재하지 않는
  용어를 지어냄. `SYSTEM_PROMPT`에 "sTIL 정의를 임의로 풀어쓰지 말 것" 같은
  제약을 추가해야 함 — 의료 보고서라 이런 오류는 반드시 잡고 넘어갈 것.
