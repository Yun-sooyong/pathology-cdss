# segmentation-mosec 저장소에 얹을 것 (박검모 → 서정은)

MedGemma 판독 보고서 라우트를 GPU 서비스에 추가하기 위한 인수인계 문서.

**역할 분담은 파일 경계로 나눕니다.** 같은 파일을 "뼈대/로직"으로 쪼개면 결국
둘 다 그 파일을 건드리게 되어 충돌합니다.

| 파일 | 소유 | 상태 |
|---|---|---|
| `src/medgemma_worker.py` | **박검모** (새 파일) | ✅ 작성 완료 — 그대로 복사 |
| `src/server.py` | 서정은 | 🔴 아래 2번 적용 필요 |
| `Dockerfile` | 서정은 | 🔴 아래 3번 적용 필요 |
| `requirements.txt` | 서정은 | 🔴 아래 4번 적용 필요 |
| `src/config.py` | 서정은 | 🔴 아래 5번 적용 필요 |

기존 파일은 제가 건드리지 않았습니다. 아래 내용을 정은님이 적용해 주세요.

---

## 1. `src/medgemma_worker.py` — 새 파일 (그대로 복사)

이 폴더의 `medgemma_worker.py`를 `src/` 아래에 두면 됩니다.
프롬프트를 이 파일에서 만들지 않는다는 점만 유의해 주세요 — `system`/`text` 모두
호출자(cdss-worker)가 완성해서 보냅니다. sTIL 용어 할루시네이션을 막는 제약이
그 프롬프트에 들어있어서, 서비스 쪽에서 덧붙이거나 갈아끼우면 방어가 사라집니다.

## 2. 🔴 `src/server.py` — `append_worker` → `register_runtime` 마이그레이션

**여기가 제일 중요한 부분입니다.** 현재 `build_server()`는 `append_worker`로
**단일 파이프라인**을 만들고 있어서, 워커를 하나 더 붙이면 `/medgemma`가 생기는 게
아니라 **모든 세그멘테이션 요청이 MedGemma까지 통과**하게 됩니다.

라우트를 나누려면 `register_runtime`으로 바꿔야 합니다.

```python
from mosec import Runtime, Server          # Runtime 추가 임포트
from .medgemma_worker import MedGemmaReport


def build_server() -> Server:
    server = Server()

    preprocess_rt = Runtime(Preprocess, num=config.NUM_PREPROCESS_WORKERS)

    inference_env = [{"CUDA_VISIBLE_DEVICES": config.CUDA_VISIBLE_DEVICES}] * config.NUM_INFERENCE_WORKERS
    inference_rt = Runtime(
        Inference,
        num=config.NUM_INFERENCE_WORKERS,
        max_batch_size=config.MAX_BATCH_SIZE,
        max_wait_time=config.MAX_WAIT_TIME_MS,
        env=inference_env,
    )

    # 생성은 케이스마다 길이가 제각각이라 배치 이득이 적어 batch 없이 둡니다.
    medgemma_env = [{"CUDA_VISIBLE_DEVICES": config.CUDA_VISIBLE_DEVICES}] * config.NUM_MEDGEMMA_WORKERS
    medgemma_rt = Runtime(
        MedGemmaReport,
        num=config.NUM_MEDGEMMA_WORKERS,
        env=medgemma_env,
    )

    server.register_runtime({
        "/inference": [preprocess_rt, inference_rt],
        "/medgemma": [medgemma_rt],
    })
    return server
```

> ⚠️ `/inference` 경로가 유지되는지 꼭 확인해 주세요. 워커(cdss-worker)가 이미
> `POST /inference`로 호출하고 있어서, 경로가 바뀌면 세그멘테이션도 같이 끊깁니다.
>
> mosec 버전이 `register_runtime`을 지원하지 않으면 알려주세요 — 그때는 라우트
> 분리 대신 별도 서비스로 가야 하는지 다시 논의해야 합니다.

## 3. 🔴 `Dockerfile` — MedGemma 가중치를 빌드타임에 포함

런타임에 HuggingFace를 부르지 않도록 이미지에 구워 넣습니다.
`google/medgemma-4b-it`는 **gated 모델이라 빌드 시 토큰이 필요**합니다.

```dockerfile
ENV HF_HOME=/opt/hf
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" \
    python -c "from huggingface_hub import snapshot_download; \
snapshot_download('google/medgemma-4b-it')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
```

빌드 명령: `docker build --secret id=hf_token,env=HF_TOKEN ...`

- ⚠️ **`ENV`/`ARG`로 토큰을 넘기면 이미지 레이어에 남습니다.** 반드시 `--secret`을 쓰세요.
- ⚠️ `--mount=type=secret`은 **BuildKit 기능**이라 `gcloud builds submit` 기본 빌더로는
  안 됩니다. `cloudbuild.yaml`에서 BuildKit을 켜거나 별도 빌드 단계가 필요합니다.
- ⚠️ 이미지가 **8GB 이상 커집니다.** Artifact Registry 용량과 빌드 시간, 그리고
  콜드스타트(min-instances=0이면 매번 발생) 영향을 확인해 주세요.

## 4. 🔴 `requirements.txt`

```
transformers
accelerate
# VRAM이 빠듯해 8bit로 줄일 때만 (MEDGEMMA_LOAD_IN_8BIT=1)
# bitsandbytes
```

## 5. 🔴 `src/config.py`

```python
NUM_MEDGEMMA_WORKERS = int(os.environ.get("NUM_MEDGEMMA_WORKERS", "1"))
```

L4 1장에 세그 모델과 MedGemma가 함께 올라가므로 **1로 시작**하는 걸 권합니다.

---

## 호출 규약 (cdss-worker ↔ /medgemma)

세그멘테이션과 동일하게 **msgpack + ID 토큰**입니다.
워커 호출부(`medgemma_report.call_medgemma()`)는 이미 이 규격으로 구현·테스트 완료라,
`medgemma_worker.py`를 그대로 쓰시면 양쪽이 맞습니다.

```
POST /medgemma
  Content-Type: application/msgpack
  Authorization: Bearer <ID 토큰>
  body: msgpack{job_id, system, text, image(PNG bytes), max_new_tokens}
  resp: msgpack{job_id, generated_text}
```

---

## 🔴 별건이지만 이게 더 급합니다 — `MC_DROPOUT_N`을 25 → 20으로

`config.MC_DROPOUT_N`이 현재 **25**인데, conformal 보정상수 `q=19.63`은
**20회 기준으로 산출된 값**입니다 (최일운 「sTIL 서빙 모델 최종 결과 보고」 6절:
"MC Dropout 횟수 — 기획안 25회 → **최종 20회** (q=19.63이 20회 기준)").

25회 결과에 그 q를 그대로 쓰면 예측구간이 겉보기엔 정상인데 **커버리지 보장이
깨집니다.** 임상 컷오프 판정이 이 구간으로 갈리는 값이라, 워커는 25회 응답을 받으면
에러로 막도록 해뒀습니다. **지금 상태로는 파이프라인이 여기서 멈춥니다.**

env var 한 줄 수정이 q 재보정보다 훨씬 쌉니다. MedGemma 라우트보다 이게 먼저예요.

## 확인 부탁드릴 것

1. **VRAM 여유** — L4 24GB에 세그 U-Net + MedGemma 4B(BF16 ~8GB) 동거 가능한지.
   빠듯하면 `MEDGEMMA_LOAD_IN_8BIT=1`로 ~4GB까지 줄일 수 있습니다.
   다만 **양자화하면 제가 잡아둔 할루시네이션 방어를 재검증**해야 하니 미리 알려주세요.
2. **가장자리 타일 패딩** — 워커는 512 배수가 아닌 가장자리 타일을 zero-padding해서
   보내고 돌아온 마스크를 원래 크기로 잘라 씁니다. `run_mc_dropout_tiled()`가
   비정사각형 입력을 그대로 받는다면 패딩을 빼는 게 나으니 알려주세요.
