# 워커 + MedGemma 인터페이스 (담당자 E: 박검모)

팀원 연동용 문서. "내가 무엇을 받고, 무엇을 주는가"를 정의한다.
발표 15·16·17번 슬라이드 기준.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `main.py` | Cloud Run Job 진입점 — `JOB_PAYLOAD` 환경변수를 읽어 `run_job()` 호출 |
| `worker.py` | 케이스 1건 처리 흐름 + Job 테이블 업데이트 |
| `segmentation_client.py` | GPU `/inference` 호출 (경로 전달) → `mc_stack` 수신 |
| `til_conformal.py` | sTIL + Conformal + 클래스 구성비 (순수 계산) |
| `medgemma_report.py` | MedGemma `/medgemma` HTTP 호출 + 5절 JSON 조립 + 재시도/폴백 |
| `Dockerfile` | CPU 전용 경량 이미지 (모델 가중치·CUDA 없음) |
| `test_*.py` | 42개 테스트 (계산 18 · 워커 14 · 세그 5 · MedGemma 5) |

## 배포

```bash
gcloud run jobs deploy cdss-worker \
  --source . --region asia-northeast3 \
  --set-env-vars DJANGO_API_BASE=https://bcss-api-XXXXXXXXXX.asia-northeast3.run.app/api \
  --memory 1Gi \
  --task-timeout 3600 --max-retries 3
```
Job 이름 `cdss-worker` / 리전 `asia-northeast3`(서울) — Django가 이 값으로
`:run` 엔드포인트를 조립하므로 바꾸면 안 됩니다.

### 🔴 배포 전 IAM 확인 (윤수용) — 이거 없으면 런타임에 실패합니다

워커가 쓰는 서비스 계정에 아래 권한이 있어야 합니다. 배포 자체는 성공하는데
**실행 시점에 403으로 죽기 때문에** 미리 확인해야 합니다.

| 권한 | 대상 | 왜 필요한가 |
|---|---|---|
| `roles/run.invoker` | `segmentation-mosec` (싱가포르) | ID 토큰으로 `/inference`·`/medgemma` 호출. **서비스·리전이 달라서 자동으로 안 붙습니다** |
| `roles/storage.objectCreator` | `...-data` 버킷 | `results/` 세그 이미지 쓰기 |

> 워커는 이제 원본을 읽지 않으므로 `objectViewer`가 필요 없습니다.
> 대신 **GPU 서비스 SA에 `uploads/` 읽기 권한**이 필요합니다.

```bash
# 예: 워커 SA에 GPU 서비스 호출 권한 부여
gcloud run services add-iam-policy-binding segmentation-mosec \
  --region asia-southeast1 \
  --member serviceAccount:<워커-SA> \
  --role roles/run.invoker
```

Django API 호출은 현재 인증 없이 PATCH하고 있습니다 — `bcss-api`가 비공개면
거기도 `run.invoker`가 필요하고, 워커 쪽에 토큰 부착 코드를 추가해야 합니다.
**공개/비공개 여부를 알려주세요.**

## 🔴 서정은 (segmentation-mosec / GPU 서빙) 에게 요청 — 내가 받는 입력

**✅ 2026-07-27 확정 — 타일 분할은 GPU 서비스가 한다. 워커는 경로만 넘긴다.**

워커가 미리 자르면 **조직이 이미지 중앙에만 있는 경우 가장자리 타일이 전부
"조직 없음"으로 거부되어 케이스 전체가 실패**합니다. 서비스가 원본 전체를 보고
조직 타일을 골라내야 그 판정이 제대로 됩니다.

이미지 바이트도 싣지 않습니다 — Cloud Run 요청 32MB 제한인데 원본은 최대 141MB
(151장 중 36%가 초과)라 인라인 전송이 불가능합니다. **경로만 보내면 제한이 사라집니다.**

```
POST {SEGMENTATION_URL}/inference
  Content-Type: application/msgpack
  Authorization: Bearer <ID 토큰>     # 비공개 서비스
  body: msgpack{job_id, gcs_image_path, model_name}     ← image 바이트 아님
  resp: msgpack{job_id, model_name_used, mc_samples, height, width, num_classes, masks(npz)}
```

**🔴 서정은 확인 필요:**
1. **요청 필드 변경** — `image`(PNG bytes) → **`gcs_image_path`**(`gs://...` 문자열).
   서비스가 GCS에서 직접 읽도록 `Preprocess` 단계를 바꿔주셔야 합니다.
2. **GPU 서비스 SA에 GCS 읽기 권한 필요** — `roles/storage.objectViewer` on `...-data`.
   워커는 이제 원본을 읽지 않으므로 반대로 워커 쪽 read 권한은 불필요해집니다.
3. **응답은 원본 해상도 기준 `(N, H, W)`** — 타일 재조립까지 마친 상태로 주세요.
   워커는 받은 배열을 그대로 sTIL 계산에 넣습니다.
4. **타임아웃** — 워커는 요청당 최대 30분(`SEGMENTATION_TIMEOUT=1800`) 대기합니다.
   실측 7.6초/타일 × 약 30타일(2529×3028) ≈ 4분이라 여유가 있지만, 더 큰 이미지가
   들어오면 Cloud Run 요청 타임아웃(최대 60분) 안에 끝나는지 확인이 필요합니다.

| 입력 | 타입 | 설명 |
|------|------|------|
| `mc_stack` (재조립 결과) | numpy `(N, H, W)` uint8 | ⚠️ 평균/argmax 하지 말고 **N개 원본 예측 그대로** (불확실도 계산에 필요) |
| `model_name` | str | 현재 보정 완료: `attention_unet_v1` **1종** (아래 🔴 참고) |

**`seg_image`는 더 이상 받지 않습니다** — 워커가 `mc_stack`의 다수결 마스크를
직접 렌더링합니다(`worker.render_seg_image()`). 구성비 수치와 MedGemma가 보는
그림이 같은 마스크에서 나오므로 서로 어긋나지 않습니다.

## 🔵 MedGemma — GPU 서비스 `/medgemma` 라우트 (**내가 만든다**)

아키텍처 문서 2.5절 기준 이 라우트의 담당은 **박검모(나)**다. 서정은에게 규격을
물어보는 게 아니라, **내가 아래 규격으로 라우트를 구현해 서정은의 GPU 서비스에
얹는 것**이다. 서정은에게 필요한 건 "라우트를 추가해도 되는지 + VRAM 여유"뿐.

```
POST {MEDGEMMA_URL}
  Content-Type: application/msgpack
  Authorization: Bearer <ID 토큰>
  body: msgpack{job_id, system, text, image(PNG bytes), max_new_tokens}
  resp: msgpack{job_id, generated_text}
```
세그멘테이션과 같은 서비스라 **msgpack + ID 토큰** 규약을 그대로 따릅니다
(base64가 아니라 원본 PNG 바이트 — 33% 오버헤드 없음).

워커 호출부(`medgemma_report.call_medgemma()`)는 이미 이 규격으로 구현돼 있으므로,
GPU 쪽 라우트만 같은 규격으로 만들면 양쪽이 맞는다.

**🔴 서정은에게 확인할 것 (규격이 아니라 수용 가능 여부):**
- L4 24GB에 세그 U-Net + MedGemma 4B(BF16 ~8GB)를 함께 올릴 때 VRAM 여유가 있는지
- mosec에 두 번째 파이프라인(라우트)을 등록하는 방식 — 서비스 코드 구조 공유 필요
- VRAM이 빠듯하면 int8 양자화 필요 → **양자화 시 sTIL 할루시네이션 재검증 필수**

**⚠️ 프롬프트는 워커가 소유합니다.** `SYSTEM_PROMPT`에 sTIL 용어 할루시네이션을
막는 제약이 들어있고(2026-07-24 실측으로 발견해 추가), 응답 파싱·재시도·폴백도
워커 쪽에 있습니다. **GPU 서비스에서 프롬프트를 자체적으로 만들어 덮어쓰지 마세요** —
그러면 할루시네이션 방어가 사라집니다. 서비스는 받은 문자열을 그대로 모델에 넣고
생성 결과를 그대로 돌려주기만 하면 됩니다.

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

**✅ 2026-07-24 실제 데이터로 검증 완료** — 모델 학습 쪽(최일운)의 실제 U-Net 출력 npy
(`unet_TCGA-A2-A3XX-DX1_..._MPP-0.2500.npy`)로 확인:
- `shape (20, 2529, 3028)`, `dtype uint8`, `unique [0 1 2 3 4]` — 위 기대 스펙과 완전히 일치
- `compute_stil_and_conformal()`에 그대로 넣어서 `stil_value=4.75, decision="LOW"` 정상 산출
- 포맷 불일치 리스크는 해소됨. 다만 **컷오프(50%) 근처 케이스에서 abstain=True가 뜨는지는 아직 미검증** (이번 샘플은 4.75%라 쉬운 케이스였음) — 근처 값 나는 다른 슬라이드로 한 번 더 확인 필요

**✅ 전달 방식 확정:** 워커는 경로만 전달, GPU 서비스가 GCS에서 직접 읽고 타일링 (위 참조).

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
2. ✅ **확정 (2026-07-27)** — **페이로드**: 환경변수 `JOB_PAYLOAD`(JSON)로 키 4개.
   `{"job_id", "model_name", "gcs_image_path", "gcs_result_path"}`.
   세그 결과 파일은 받지 않고 워커가 원본을 읽어 GPU를 직접 호출하며,
   렌더링한 세그 이미지를 `gcs_result_path`에 씁니다. `main.py`가 필수 키를 검사합니다.
3. ✅ **해소 (2026-07-27)** — **MedGemma Docker**: MedGemma가 GPU 서비스로 이동하여
   워커 이미지에서 가중치·CUDA가 모두 빠졌습니다. 워커는 CPU 전용 경량 이미지입니다.
4. ✅ **확정 (2026-07-24)** — **상태 전이**: `status="sTIL 완료"` → MedGemma 후 `status="완료"` 규약대로
   `run_job()`에 두 번의 `update_job()` 호출로 구현 완료.
5. **Cloud Run Job 메모리**: `(20, 2529, 3028)` 기준 워커 실측 피크 RSS **약 0.4GB**.
   여기에 원본 이미지(최대 141MB)와 타일 응답 버퍼가 더해지므로 **1GB 이상** 권장.
   모델은 안 올라가므로 GPU/대용량 메모리는 불필요합니다.

## ⚠️ 아직 placeholder (실제 값으로 교체 필요)

- ✅ **해소 (2026-07-26)** — `CONFORMAL_Q`가 실제 calibration 값으로 교체됨.
  `attention_unet_v1 = 19.63` (CV+, 비정규화, α=0.10, MC Dropout 20회 기준).
  Held-out 6개 기관 45 ROI에서 커버리지 95.6%로 목표 90% 충족 확인.
  ⚠️ **q는 sigma 배수가 아니라 구간 절반폭(%p) 자체**이며, MC 20회 조건에 묶여 있음 —
  MC 횟수를 바꾸면 q를 다시 보정해야 함.
- **MedGemma 양자화 판단은 GPU 서비스 쪽으로 넘어감** — 워커가 모델을 들고 있지 않으므로
  int8 양자화 여부는 GPU 서빙 담당(서정은)이 결정합니다. 다만 **양자화하면 sTIL 할루시네이션 재검증이
  필요**하다는 점은 전달해야 합니다 (정밀도가 바뀌면 생성 결과가 달라짐).
- `findings` 프롬프트 품질 — 아직 입력값 재진술 경향. 튜닝 예정.
- **🔴 할루시네이션 발견 (2026-07-24)**: 실제 npy로 테스트 중 MedGemma가
  `segmentation_summary`에서 "sTIL(시스플라스크 림프관 세포)"라는 존재하지 않는
  용어를 지어냄. `SYSTEM_PROMPT`에 "sTIL 정의를 임의로 풀어쓰지 말 것" 같은
  제약을 추가해야 함 — 의료 보고서라 이런 오류는 반드시 잡고 넘어갈 것.
