# 워커(Conformal + sTIL) + MedGemma 판독 보고서 — Cloud Run Job 이미지
#
# 발표 16번 슬라이드 방침: MedGemma 가중치를 **빌드타임에 이미지에 포함**한다.
# 런타임에 HuggingFace를 호출하지 않으므로 외부 네트워크 의존이 없고,
# 콜드스타트 때 8GB를 내려받느라 지연되는 일도 없다.
#
# ⚠️ 그 대가로 이미지가 매우 커진다 (BF16 4B ≈ 8GB + 베이스/의존성).
#    빌드 시간과 Artifact Registry 저장 비용을 인프라 담당과 확인할 것.
#    이미지를 줄여야 하면 medgemma_report.load_pipeline()의 int8 양자화 블록을
#    켜는 것이 아니라(그건 런타임 메모리 절감), 빌드 단계에서 양자화된 가중치를
#    받도록 바꿔야 한다.

FROM python:3.11-slim

# torch/transformers가 요구하는 최소 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성을 먼저 설치해 레이어 캐시를 살린다 (코드만 바뀌면 재설치 안 함)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── MedGemma 가중치를 빌드타임에 내려받아 이미지에 굽는다 ──────────────────
# medgemma-4b-it는 gated 모델이라 빌드 시 토큰이 필요하다.
#   docker build --secret id=hf_token,env=HF_TOKEN ...
# ⚠️ ENV/ARG로 토큰을 넘기면 이미지 레이어에 남는다. 반드시 --secret을 쓸 것.
ENV HF_HOME=/opt/hf
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" \
    python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('google/medgemma-4b-it')"

# 런타임에 바깥으로 나가지 않도록 오프라인 고정
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# ── 워커 코드 ─────────────────────────────────────────────────────────────
COPY til_conformal.py medgemma_report.py worker.py main.py ./

# Django API 주소는 배포 환경에서 주입 (기본값은 자리표시자라 그대로 두면 실패한다)
ENV DJANGO_API_BASE=""

ENTRYPOINT ["python", "main.py"]
