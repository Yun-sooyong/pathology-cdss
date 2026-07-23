# BCSS CDSS — 유방암 병리 이미지 CDSS 프로젝트

BCSS(Breast Cancer Semantic Segmentation) 데이터셋 기반 조직 세그멘테이션, TIL(종양침윤림프구) 비율 계산, Conformal Prediction 기반 판정 유보(deferral) 메커니즘을 제공하는 임상 의사결정 지원 시스템(CDSS)입니다.

## 디렉토리 구조

| 폴더 | 담당 | 내용 |
| --- | --- | --- |
| `backend/` | 윤수용 | Django REST API, 인프라 (Cloud Run, Cloud SQL, Cloud Tasks 연동) |
| `model-training/` | 담당자 B | Attention U-Net 학습 파이프라인, ONNX 변환 |
| `serving/segmentation-mosec/` | 담당자 C | 세그멘테이션 Mosec 서비스 (MC Dropout 배치 추론, GPU) |
| `serving/medgemma-mosec/` | 담당자 E | MedGemma Mosec 서비스 (자체 호스팅 VLM, GPU) |
| `worker/` | 담당자 E | Cloud Tasks 워커 — Conformal Prediction, sTIL 비율 계산, 판독 보고서 조립 |
| `frontend/` | 담당자 D | React + Vite 프론트엔드 |
| `docs/` | 공통 | API 스펙, 공통 인터페이스 문서 (수정 전 팀 전체 공지 필수) |

## 브랜치 전략

- `main` 브랜치는 보호(protected) — 직접 push 금지
- 각자 `feature/이름` 브랜치에서 작업 후 PR로 병합 (예: `feature/sooyong`, `feature/gummo`, `feature/ilun`)
- PR은 원칙적으로 **본인 담당 폴더만** 수정 — merge 충돌 최소화
- `docs/` 하위 공통 파일을 수정해야 하는 PR은 제목에 명시하고 팀 채널(`medi_ai_group_3@googlegroups.com`)에 공지

## 공통 인터페이스

파트 간 연동 규격(Job 테이블 스키마, API 요청/응답 형식, medgemma_report 필드 등)은 [`docs/api-spec.md`](./docs/api-spec.md)를 기준으로 한다. 변경 시 반드시 팀 전체에 공지.

## 리전

- Django, Cloud Tasks, 워커, Cloud SQL → 서울 (`asia-northeast3`)
- 세그멘테이션 Mosec, MedGemma Mosec (GPU) → 도쿄 (`asia-northeast1`) — Cloud Run GPU 미지원 리전이라 분리

## GCP 프로젝트

`model-platform-502402`
