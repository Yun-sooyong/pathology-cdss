# 파이프라인 연동 규격 (A방식 확정)

**작성:** 윤수용 (backend/인프라) · **작성일:** 2026-07-27
**목적:** 브랜치가 흩어져 있어 서로의 규격을 모르는 상태를 정리한다. 각자 할 일을 확정한다.

---

## 1. 확정된 구조 — A방식

GPU 서비스는 **1개**로 유지하고, MedGemma를 **같은 mosec 서버의 다른 라우트**로 추가한다.
무거운 오케스트레이션(타일링·재조립·Django 통신)은 **CPU 워커**가 맡는다.

```
[프론트] ──업로드──> [Django, 서울]
                        │  Signed URL 발급 → 프론트가 GCS에 직접 PUT
                        │  Cloud Tasks 등록
                        ↓
                   [워커 Job, CPU, 서울]
                        ├─ ① GCS에서 원본 이미지 다운로드
                        ├─ ② 512x512 타일로 분할
                        ├─ ③ POST /inference   (타일마다) ──┐
                        ├─ ④ masks 재조립 → mc_stack       │  [GPU 서비스, 싱가포르]
                        ├─ ⑤ conformal + sTIL 계산 (numpy)  │  segmentation-mosec
                        ├─ ⑥ POST /medgemma  ───────────────┘  라우트 2개
                        └─ ⑦ PATCH Django (결과 기록)
```

**왜 A인가**
- GPU 서비스가 1개로 유지된다 (비용·할당량 부담 최소)
- 워커가 CPU라 서울에 둘 수 있고 이미지가 가볍다 (수백 MB)
- `mc_stack`(146MB)을 GCS에 올렸다 내리는 왕복이 사라진다 — 워커 메모리에서 처리
- 각 구성요소의 역할이 선명하다 (GPU=추론, CPU=조립·통신)

---

## 2. 이미 완료된 것 (윤수용)

### 2.1 Django API — 배포 완료

**Base URL:** `https://bcss-api-XXXXXXXXXX.asia-northeast3.run.app`

| 엔드포인트 | 용도 | 상태 |
|---|---|---|
| `POST /api/batches` | 배치 생성 + GCS 서명 업로드 URL 발급 | ✅ |
| `POST /api/batches/{id}/start` | 업로드 완료 → Cloud Tasks 등록 | ✅ |
| `GET /api/result/{batch_id}` | 요약(총/완료/에러) + job 리스트 | ✅ |
| `GET /api/jobs/{job_id}` | 개별 결과 조회 | ✅ |
| `PATCH /api/jobs/{job_id}/` | **워커가 결과 기록** | ✅ |

검증 완료: 141MB 파일 GCS 직접 업로드 3.6초, Vercel 도메인에서 CORS 통과.

### 2.2 워커 호출 규격 — 검모님 `main.py`에 맞춤 완료

Cloud Tasks가 Cloud Run Admin API로 Job을 실행시키며, 아래 환경변수를 주입한다.

```
JOB_PAYLOAD = '{"job_id":"...","model_name":"attention_unet_v1","gcs_image_path":"gs://..."}'
```

`worker_dev/main.py`의 `read_payload()`가 그대로 읽는다.

> ⚠️ `mc_stack_path` / `gcs_result_path`는 **보내지 않는다.**
> 그 파일을 만드는 주체가 없기 때문이며, A방식에서는 워커가 segmentation을
> 직접 호출하므로 불필요하다. (`run_job()` 수정 필요 — 4.2 참고)

### 2.3 결과 기록 규격 — 검모님 `worker.py`에 맞춤 완료

`worker.update_job()`이 보내는 형식을 **그대로** 받도록 Django를 수정했다.
**워커 코드는 이 부분을 고칠 필요가 없다.**

```
PATCH {DJANGO_API_BASE}/jobs/{job_id}/
```

| 워커가 보내는 값 | Django 처리 |
|---|---|
| `status="sTIL 완료"` | → 내부 키 `stil_done` 으로 저장 (한글 라벨 그대로 수용) |
| `status="완료"` | → `done` |
| `stil_value` (float) | 그대로 저장 |
| `cp_value` (float) | `conformal_prediction_value` JSON에 `{"cp_value": ...}` 로 저장 |
| `lower`/`upper`/`abstain`/`decision`/`sigma` | 함께 보내면 같은 JSON에 병합 |
| `medgemma_report` (dict) | 그대로 저장 |

**배포 시 워커에 주입할 값:**
```
DJANGO_API_BASE=https://bcss-api-XXXXXXXXXX.asia-northeast3.run.app/api
```
(끝에 `/api` 포함 — 워커가 `f"{BASE}/jobs/{id}/"` 로 조립하므로)

---

## 3. GPU 서비스 (서정은) — 할 일

### 3.1 현재 상태 (확인 완료)

`segmentation-mosec` · `asia-southeast1`(싱가포르) · L4 GPU 1 · 4 CPU / 16Gi
`FORCE_DUMMY=0`, `MC_DROPOUT_N=25`, `TILE_SIZE=512`, concurrency 4

**실측 검증:** 512×512 타일 1개 → **HTTP 200, 7.6초**, `mc_samples=25`, mask 512×512, 5 classes ✅
진짜 모델로 정상 동작 중.

### 3.2 요청사항 — `/medgemma` 라우트 추가

mosec은 **멀티 라우트를 지원한다** (`register_runtime`이 `Dict[str, List[Runtime]]`을 받음).

```python
server.register_runtime({
    "/inference": [Runtime(Preprocess, num=...), Runtime(Inference, num=..., max_batch_size=...)],
    "/medgemma":  [Runtime(MedGemmaWorker, num=1)],   # ← 신규 (검모님 코드 이관)
})
```

- L4 24GB VRAM: 세그 U-Net(수백 MB) + MedGemma 4B BF16(~8GB) → **용량 문제 없음**
- MedGemma 로직은 검모님 `medgemma_report.py`를 이관

### 3.3 확인 부탁 — 서비스가 비공개다

토큰 없이 호출하면 **401**이 난다. 워커가 호출하려면:
- 워커 서비스 계정에 `roles/run.invoker` 권한 필요 (인프라에서 부여 — 5.2)
- 워커 코드가 **ID 토큰**을 붙여야 함 (4.3 참고)

---

## 4. 워커 (박검모) — 할 일

### 4.1 페이로드 읽기 — 그대로 두면 됨 ✅

`main.py`의 `JOB_PAYLOAD` 방식 그대로 동작한다. 단, 키는 아래 3개만 온다.

```python
{"job_id": "...", "model_name": "...", "gcs_image_path": "gs://..."}
```

### 4.2 `run_job()` 수정 — mc_stack을 직접 만든다

현재:
```python
mc_stack  = load_mc_stack(payload["mc_stack_path"])      # ❌ 그 파일이 없음
seg_image = load_seg_image(payload["gcs_result_path"])   # ❌
```

변경:
```python
image     = download_gcs_blob(payload["gcs_image_path"])  # 원본 다운로드
tiles     = split_tiles(image, 512)                       # 타일 분할
mc_stack  = call_segmentation(tiles, model_name)          # 호출 후 재조립
seg_image = render_seg_image(mc_stack)                    # MedGemma 입력용
```

**왜 타일 분할이 필요한가:** Cloud Run 요청 크기 제한이 **32MB**다.
실측 이미지는 최대 **141MB**(151장 중 36%가 32MB 초과)라 원본을 통째로 못 보낸다.
서비스 설정도 `MOSEC_MAX_REQUEST_SIZE=40MB`로 잡혀 있다.

### 4.3 segmentation 호출 규격 (실측 검증 완료)

```python
POST https://segmentation-mosec-xxxxxxxxxx-as.a.run.app/inference
Headers:
  Content-Type: application/msgpack
  Authorization: Bearer <ID 토큰>          # 비공개 서비스라 필수
Body: msgpack.packb({"job_id": ..., "image": <PNG bytes>, "model_name": ...}, use_bin_type=True)

응답: msgpack → {"job_id", "model_name_used", "mc_samples", "height", "width",
                 "num_classes", "masks"(npz 압축 bytes)}
```

Cloud Run Job 안에서 ID 토큰 얻는 법:
```python
import google.auth.transport.requests, google.oauth2.id_token
req = google.auth.transport.requests.Request()
token = google.oauth2.id_token.fetch_id_token(req, SEGMENTATION_URL)
```

### 4.4 MedGemma는 HTTP 호출로 변경

현재 `run_job(payload, pipe)`가 파이프라인을 **in-process로 로드**한다.
A방식에서는 GPU 서비스의 `/medgemma` 라우트를 호출하므로 `pipe` 인자가 불필요하다.
→ 워커는 **CPU만으로 동작**하고 이미지가 가벼워진다(수백 MB).

### 4.5 Dockerfile 경량화

현재 `worker_dev/Dockerfile`은 MedGemma 8GB를 빌드타임에 굽는다. A방식에서는 불필요하다.

또한 현재 방식은 그대로는 빌드가 안 된다:
- `RUN --mount=type=secret` 은 **BuildKit 기능**이라 `gcloud run jobs deploy --source` 기본 빌드가 지원하지 않는다 → 별도 `cloudbuild.yaml` 필요
- `huggingface_hub` 가 `requirements.txt` 에 없다
- 베이스가 `python:3.11-slim`이라 **CUDA가 없다** → `device_map="auto"`가 조용히 CPU로 폴백될 위험
  (정은님이 겪은 것과 같은 함정)

A방식에서는 위 문제가 **전부 사라진다** (MedGemma를 워커가 안 들고 있으므로).

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .           # numpy, pillow, requests, google-cloud-storage, msgpack
RUN pip install --no-cache-dir -r requirements.txt
COPY til_conformal.py worker.py main.py ./
ENTRYPOINT ["python", "main.py"]
```

### 4.6 배포

```bash
gcloud run jobs deploy cdss-worker \
  --source . --region asia-northeast3 \
  --set-env-vars DJANGO_API_BASE=https://bcss-api-XXXXXXXXXX.asia-northeast3.run.app/api \
  --task-timeout 3600 --max-retries 3
```

> Job 이름은 **`cdss-worker`**, 리전은 **`asia-northeast3`(서울)** 로 맞춰주세요.
> Django가 이 이름/리전으로 `:run` 엔드포인트를 조립합니다.
> 다르게 하실 거면 알려주시면 Django 설정(`WORKER_JOB_NAME`/`WORKER_JOB_REGION`)을 바꾸겠습니다.

---

## 5. 인프라 (윤수용) — 할 일

### 5.1 완료
- Cloud Tasks 큐 `inference-queue` (서울) — 등록 정상, OAuth 인증 통과
- GCS 버킷 CORS, 서명 URL용 IAM(`serviceAccountTokenCreator`)
- `actAs` / `run.developer` 권한

### 5.2 남음 — 워커가 GPU 서비스를 호출할 권한
```bash
gcloud run services add-iam-policy-binding segmentation-mosec \
  --region=asia-southeast1 \
  --member="serviceAccount:1036429513149-compute@developer.gserviceaccount.com" \
  --role="roles/run.invoker" --project=model-platform-502402
```

### 5.3 큐 정리 (워커 배포 직전 1회)
오래된 더미 태스크 14개가 재시도 중이다.
```bash
gcloud tasks queues purge inference-queue --location=asia-northeast3 --quiet
```

---

## 6. 🔴 팀 결정 필요 — `unet_v1` 문제

검모님 `worker.py`는 보정되지 않은 모델을 **의도적으로 차단**한다 (타당한 안전장치).

```python
CONFORMAL_Q = {"attention_unet_v1": 19.63}   # unet_v1 없음 → 예외 발생
```

그런데 **Django·프론트의 기본 모델이 `unet_v1`** 이라, 지금 상태로는 워커가 100% 실패한다.
일운님 서빙 모델도 `attention_unet_v1.onnx`이므로 그쪽이 정본으로 보인다.

**제안:** 프론트 기본값·선택지를 `attention_unet_v1` 하나로 정리한다.
(`unet_v1`은 calibration 후 다시 노출)
→ 결정되면 Django의 `MODEL_CHOICES`도 함께 정리하겠습니다.

---

## 7. 진행 순서 제안

MedGemma는 HF 게이트 승인·토큰이 필요해 시간이 걸린다. **2단계로 나눈다.**

**1단계 (오늘 가능)** — MedGemma 없이 파이프라인 완주
- 워커: 세그 호출 + 타일 재조립 + conformal/sTIL + `status="sTIL 완료"` PATCH
- 화면에 `stil_value`·`conformal_prediction_value` 실제 값 표시
- GPU/게이트 모델 무관, CPU 워커만 있으면 됨

**2단계** — MedGemma 추가
- GPU 서비스에 `/medgemma` 라우트
- 워커가 호출 → `medgemma_report` 채우고 `status="완료"`

---

## 8. 현재 배포 현황

| 구성요소 | 리전 | 상태 |
|---|---|---|
| `bcss-api` (Django) | 서울 | ✅ 배포·검증 완료 |
| `segmentation-mosec` (GPU) | 싱가포르 | ✅ 실모델 동작 확인 (7.6초/타일) |
| 프론트 (Vercel) | — | 🟡 코드 완료, **재배포 필요** |
| **워커** | 서울 | ❌ **미배포** ← 현재 병목 |
| MedGemma | 싱가포르 | ❌ 2단계 |
