# Welcome to GitHub Desktop!

This is your README. READMEs are where you can communicate what your project is and how to use it.

Write your name on line 6, save it, and then head back to GitHub Desktop.
gmoo

| 폴더 | 담당 | 내용 |
| --- | --- | --- |
| `backend/` | 윤수용 | Django REST API, 인프라 (Cloud Run, Cloud SQL, Cloud Tasks 연동) |
| `model-training/` | 최일운 | Attention U-Net 학습 파이프라인, ONNX 변환 |
| `serving/segmentation-mosec/` | 서정은 | 세그멘테이션 Mosec 서비스 (MC Dropout 배치 추론, GPU) |
| `serving/medgemma-mosec/` | 박검모 | MedGemma Mosec 서비스 (자체 호스팅 VLM, GPU) |
| `worker/` | 박검모 | Cloud Tasks 워커 — Conformal Prediction, sTIL 비율 계산, 판독 보고서 조립 |
| `frontend/` | 정다희 | React + Vite 프론트엔드 |
| `docs/` | 공통 | API 스펙, 공통 인터페이스 문서 (수정 전 팀 전체 공지 필수) |
## Claude Code 오케스트레이션 세팅

이 레포는 Claude Code 워크플로우/서브에이전트/Codex 플러그인 세팅의 원본입니다.
새 컴퓨터 세팅 방법은 [SETUP.md](SETUP.md) 참고 — 클론 후 `setup.ps1`(Windows) 또는 `setup.sh`(macOS/Linux) 실행.
