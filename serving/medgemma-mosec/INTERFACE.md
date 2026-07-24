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
| `model_name` | str | `unet_v1` / `attention_unet_v1` |

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
  "segmentation_summary": "조직 이미지에서 유방 조직과 암세포가 혼합되어...",
  "findings": "sTIL 47%로 경계 부위에서 국소적 침윤 양상이...",
  "disclaimer": "본 결과는 AI 보조 도구의 출력이며 의료진의 최종 판단이 필요합니다."
}
```

**화면 필드 매핑:**
| React 화면 요소 | JSON 필드 |
|----------------|-----------|
| sTIL 큰 숫자 | `stil_ratio` |
| CP 값 (0.86) | `uncertainty.cp_value` |
| 예측구간 막대 | `uncertainty.lower` ~ `upper` |
| 기권 뱃지 | `uncertainty.abstain` (true면 빨간 "기권") |
| VLM 판독결과지 | `findings` |
| 요약 | `segmentation_summary` |
| 하단 주의문구 | `disclaimer` |

**협의 필요:** 발표 24번 슬라이드는 CP 값을 단일 스칼라(0.86)로 표시.
→ 나는 `cp_value`(단일) + `lower/upper`(구간) **둘 다** 제공. 화면에 뭘 쓸지 다희가 선택.

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
4. ✅ **확정 (2026-07-24)** — **상태 전이**: `status="sTIL 완료"` → MedGemma 후 `status="완료"` 규약대로
   `run_job()`에 두 번의 `update_job()` 호출로 구현 완료.

## ⚠️ 아직 placeholder (실제 값으로 교체 필요)

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
