# 업로드 용량 제한 대응 방안 (A안 / B안)

**작성:** 윤수용 (backend) · **작성일:** 2026-07-27
**상태:** 검토 요청 — 프론트(정다희) 협의 필요

---

## 1. 문제

배포된 Django API(`bcss-api`, Cloud Run)에 병리 이미지를 업로드할 때 **32MB를 넘으면 실패**한다.

원인은 우리 코드 설정이 아니라 **Cloud Run 플랫폼의 하드 리밋**이다.

> Cloud Run 공식 문서 (quotas):
> _"Maximum HTTP/1 request size: **32 MiB** if using HTTP/1 server. **No limit if using HTTP/2** server."_

- 이 값은 **할당량 증설이 불가능**하다.
- Django의 `MAX_UPLOAD_SIZE_MB = 50` 설정은 **무의미**하다. Django가 요청을 받기 전에 Cloud Run이 먼저 차단한다.

### 실측 데이터 (`BCSS_raw/images`, PNG 151장)

| 지표          | 값                          |
| ------------- | --------------------------- |
| 최대          | **141.1 MB** ← 제한의 4.4배 |
| 평균          | 31.7 MB                     |
| 중앙값        | 25.5 MB                     |
| 최소          | 2.4 MB                      |
| **32MB 초과** | **54장 (36%)**              |
| 16MB 초과     | 110장 (73%)                 |

**이미지 3분의 1 이상이 한 장만으로도 제한을 넘는다.**

---

## 2. A안 — Signed URL 직접 업로드

프론트가 **GCS에 직접** 업로드하고, Django는 경로만 관리한다.
파일이 Cloud Run을 거치지 않으므로 **용량 제한이 사라진다.**

### 흐름

```
① 프론트 → Django    POST /api/batches
                     { model_name, files: [{filename, size}, ...] }
                     ↓ Batch + Job N개 생성(status=waiting), 서명 URL 발급
② Django → 프론트    { batch_id, uploads: [{job_id, upload_url, gcs_path}, ...] }

③ 프론트 → GCS       각 파일을 upload_url로 직접 PUT   ← 용량 제한 없음

④ 프론트 → Django    POST /api/batches/{batch_id}/start
                     ↓ Cloud Tasks에 작업 등록

⑤ 조회 (기존과 동일)  GET /api/result/{batch_id}
                     GET /api/jobs/{job_id}
```

### 작업 목록

**백엔드**

| #   | 항목                                  | 내용                              |
| --- | ------------------------------------- | --------------------------------- |
| 1   | `POST /api/batches` (신규)            | Batch·Job 생성 + V4 서명 URL 발급 |
| 2   | `POST /api/batches/{id}/start` (신규) | 업로드 완료 신호 → 큐 등록        |
| 3   | `services/gcs.py`                     | `generate_upload_url()` 추가      |
| 4   | 조회 API                              | **변경 없음**                     |
| 5   | `MAX_UPLOAD_SIZE_MB`                  | 50 → 200 (또는 제거)              |

**인프라 (⚠️ 함정 2개)**

1. **GCS 버킷 CORS 설정** — 브라우저가 `storage.googleapis.com`에 직접 PUT하려면 필수.
   설정하지 않으면 프론트에서 CORS 에러가 난다.

   ```json
   [
     {
       "origin": ["https://ccdss-project.vercel.app", "http://localhost:5173"],
       "method": ["PUT", "GET"],
       "responseHeader": ["Content-Type"],
       "maxAgeSeconds": 3600
     }
   ]
   ```

2. **서명 권한** — Cloud Run 기본 서비스 계정은 **개인키가 없어서** `generate_signed_url()`이
   그냥은 실패한다. IAM SignBlob API를 쓰도록 구현하고 권한을 부여해야 한다.
   ```bash
   gcloud iam service-accounts add-iam-policy-binding \
     1036429513149-compute@developer.gserviceaccount.com \
     --member="serviceAccount:1036429513149-compute@developer.gserviceaccount.com" \
     --role="roles/iam.serviceAccountTokenCreator" \
     --project=model-platform-502402
   ```

**프론트 (정다희)**

- `uploadBatch()` 를 3단계로 교체 (URL 요청 → GCS PUT → start 호출)
- 부수 효과: 파일별 **업로드 진행률 표시**가 가능해진다 (현재 구조에선 불가)

### 장점

- **용량 제한 완전 해제** (141MB도 문제없음)
- Cloud Run 대역폭·처리시간 비용 절감
- 업로드 진행률 UI 구현 가능
- 대용량 파일 처리의 **업계 표준 패턴**

### 단점 / 주의

- 프론트·백엔드 **양쪽 변경** 필요
- 파일이 Django를 거치지 않으므로 **서버측 이미지 검증이 사라짐**
  (현재는 `ImageField`가 실제 이미지인지 검사 중)
  - 보완: 서명 URL 발급 시 **Content-Type·크기 범위 제한**을 걸어둔다
  - 보완: 워커가 다운로드 시 한 번 더 검증한다
- GCS CORS·IAM 설정이 선행되어야 하며, 빠뜨리면 원인 파악이 어렵다

---

## 3. 실행 순서

1. **인프라 선행** (윤수용)
   - GCS 버킷 CORS 설정
   - IAM `roles/iam.serviceAccountTokenCreator` 부여
   - _이 단계가 막히면 나머지가 무의미하므로 먼저 검증_
2. **백엔드 구현·배포** (윤수용)
   - `POST /api/batches`, `POST /api/batches/{id}/start`
   - `MAX_UPLOAD_SIZE_MB` 상향
3. **검증** (윤수용)
   - 서명 URL 발급 → 실제 141MB 파일 PUT 테스트
4. **프론트 규격 전달·연동** (윤수용 → 정다희)
