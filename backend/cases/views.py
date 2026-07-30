import logging 

from django.conf import settings
from django.db.models import Count
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Batch, Job
from .serializers import (
    BatchCreateSerializer,
    BatchInitSerializer,
    CaseListItemSerializer,
    BatchSerializer,
    JobSerializer,
)
from .services import gcs, tasks


logger = logging.getLogger(__name__) 
# "헬스 체크(health check)" 라는 관례에서 온 이름, 서버가 살아있고 정상 동작하는지 확인하는 용도의 엔드포인트
@api_view(["GET"])
def health(request) :
    # cloud run 체크 
    return Response({'status':'ok'})

@api_view(["POST"])
def upload(request):
    """POST /api/upload — 이미지 1~15장을 한 묶음(Batch)으로 업로드.

    흐름: 검증 → Batch 생성 → (이미지마다) GCS 업로드 → Job 생성 → 큐 등록
    응답: { "batch_id": "..." }
    """
    # 0) 검증 — 장수(1~15), 각 이미지 형식/용량, model_name
    serializer = BatchCreateSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    images = serializer.validated_data["images"]
    model_name = serializer.validated_data["model_name"]

    # 1) 묶음(Batch)을 먼저 만든다.
    batch = Batch.objects.create(model_name=model_name)

    # 2) 이미지 한 장씩 처리. 한 장이 실패해도 그 Job만 error로 두고 나머지는 계속 진행.
    for image in images:
        # 2-1) GCS 업로드
        try:
            gcs_image_path = gcs.upload_image(image, image.name)
        except Exception:
            logger.exception("GCS 업로드 실패: batch=%s file=%s", batch.id, image.name)
            # 업로드 실패한 이미지도 '에러' Job으로 남겨서 요약 개수에 잡히게 한다.
            Job.objects.create(
                batch=batch,
                gcs_image_path="",
                model_name=model_name,
                status=Job.STATUS_ERROR,
                error_message="이미지 저장 실패",
            )
            continue

        # 2-2) Job 생성 (대기 상태)
        job = Job.objects.create(
            batch=batch,
            gcs_image_path=gcs_image_path,
            model_name=model_name,
            status=Job.STATUS_WAITING,
        )

        # 2-3) 큐 등록. 실패하면 그 Job만 에러 처리.
        try:
            tasks.enqueue_inference(job.id, gcs_image_path, model_name)
        except Exception as exc:
            logger.exception("큐 등록 실패: job=%s", job.id)
            job.status = Job.STATUS_ERROR
            job.error_message = f"작업 큐 등록 실패: {exc}"
            job.save(update_fields=["status", "error_message", "updated_at"])

    # 3) 프론트엔드는 이 batch_id로 이후 진행상황/결과를 조회한다.
    return Response({"batch_id": str(batch.id)}, status=status.HTTP_201_CREATED)


@api_view(["GET"])
def case_list(request):
    """GET /api/cases — 대시보드용 최근 케이스 목록.

    쿼리 파라미터:
      limit   기본 20, 최대 100
      offset  기본 0
      status  waiting / processing / stil_done / done / error
      abstain true 면 판정 유보 건만

    응답: { count, summary{...}, results[...] }
      count   : 필터가 적용된 총 건수(페이징 전)
      summary : 필터와 무관한 전체 통계 (대시보드 상단 카드용)
    """
    try:
        limit = min(max(int(request.query_params.get("limit", 20)), 1), 100)
        offset = max(int(request.query_params.get("offset", 0)), 0)
    except ValueError:
        return Response(
            {"detail": "limit/offset은 정수여야 합니다."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    qs = Job.objects.all()  # Meta.ordering = ["-created_at"] → 최신순

    status_filter = request.query_params.get("status")
    if status_filter:
        if status_filter not in dict(Job.STATUS_CHOICES):
            return Response(
                {"detail": f"허용되지 않는 status 값입니다: {status_filter}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        qs = qs.filter(status=status_filter)

    # 판정 유보 필터 — 아직 JSON 안에 있어서 JSONField 조회로 거른다.
    if request.query_params.get("abstain", "").lower() == "true":
        qs = qs.filter(medgemma_report__uncertainty__abstain=True)

    count = qs.count()
    page = qs[offset : offset + limit]

    # 상단 카드용 통계 — 필터와 무관하게 전체 기준으로 센다.
    by_status = dict(
        Job.objects.values_list("status").annotate(n=Count("id")).values_list("status", "n")
    )
    summary = {
        "total": sum(by_status.values()),
        "waiting": by_status.get(Job.STATUS_WAITING, 0),
        "processing": by_status.get(Job.STATUS_PROCESSING, 0),
        "stil_done": by_status.get(Job.STATUS_STIL_DONE, 0),
        "done": by_status.get(Job.STATUS_DONE, 0),
        "error": by_status.get(Job.STATUS_ERROR, 0),
        "abstain": Job.objects.filter(medgemma_report__uncertainty__abstain=True).count(),
    }

    return Response({
        "count": count,
        "summary": summary,
        "results": CaseListItemSerializer(page, many=True).data,
    })


@api_view(["POST"])
def create_batch(request):
    """POST /api/batches — 배치를 만들고 파일별 GCS 서명 업로드 URL을 발급한다.

    Cloud Run은 요청 32MB 제한이 있어 큰 병리 이미지를 직접 못 받는다.
    그래서 파일은 서버를 거치지 않고 프론트가 GCS에 직접 PUT 하도록 URL만 내준다.

    요청: { "model_name": "unet_v1",
            "files": [{"filename": "a.png", "content_type": "image/png"}, ...] }
    응답: { "batch_id": ..., "uploads": [{job_id, filename, upload_url, gcs_path}, ...] }
    """
    serializer = BatchInitSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    model_name = serializer.validated_data["model_name"]
    files = serializer.validated_data["files"]

    # 1) 묶음 생성
    batch = Batch.objects.create(model_name=model_name)

    uploads = []
    for f in files:
        # 2) 저장 위치를 미리 정하고 Job row를 만들어 둔다(아직 파일은 없음).
        blob_name = gcs.build_blob_name(f["filename"])
        gcs_path = f"gs://{settings.GCS_DATA_BUCKET}/{blob_name}"

        job = Job.objects.create(
            batch=batch,
            gcs_image_path=gcs_path,
            model_name=model_name,
            status=Job.STATUS_WAITING,
        )

        # 3) 그 위치에 올릴 수 있는 서명 URL 발급
        try:
            url = gcs.generate_upload_url(blob_name, f["content_type"])
        except Exception as exc:
            logger.exception("서명 URL 발급 실패: batch=%s file=%s", batch.id, f["filename"])
            job.status = Job.STATUS_ERROR
            job.error_message = f"업로드 URL 발급 실패: {exc}"
            job.save(update_fields=["status", "error_message", "updated_at"])
            continue

        uploads.append({
            "job_id": str(job.id),
            "filename": f["filename"],
            "upload_url": url,
            "gcs_path": gcs_path,
        })

    if not uploads:
        return Response(
            {"detail": "업로드 URL을 하나도 발급하지 못했습니다."},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    return Response(
        {"batch_id": str(batch.id), "uploads": uploads},
        status=status.HTTP_201_CREATED,
    )


@api_view(["POST"])
def start_batch(request, batch_id):
    """POST /api/batches/{batch_id}/start — 업로드 완료를 알리고 추론 작업을 큐에 등록한다.

    요청(선택): { "uploaded_job_ids": ["...", ...] }
      - 생략하면 배치의 모든 대기 job을 등록한다.
      - 목록을 주면 그 job만 등록하고, 나머지는 '업로드 실패'로 표시한다.
        (일부 파일만 업로드에 실패해도 성공분은 진행할 수 있게 하기 위함)
    """
    batch = get_object_or_404(Batch, pk=batch_id)

    pending = list(batch.jobs.filter(status=Job.STATUS_WAITING))
    uploaded_ids = request.data.get("uploaded_job_ids")

    if uploaded_ids is not None:
        uploaded = {str(i) for i in uploaded_ids}
        targets = [j for j in pending if str(j.id) in uploaded]
        # 업로드되지 않은 나머지는 에러로 남겨 요약 개수에 잡히게 한다.
        for j in pending:
            if str(j.id) not in uploaded:
                j.status = Job.STATUS_ERROR
                j.error_message = "이미지 업로드 실패(또는 미완료)"
                j.save(update_fields=["status", "error_message", "updated_at"])
    else:
        targets = pending

    enqueued = 0
    for job in targets:
        try:
            tasks.enqueue_inference(job.id, job.gcs_image_path, job.model_name)
            enqueued += 1
        except Exception as exc:
            logger.exception("큐 등록 실패: job=%s", job.id)
            job.status = Job.STATUS_ERROR
            job.error_message = f"작업 큐 등록 실패: {exc}"
            job.save(update_fields=["status", "error_message", "updated_at"])

    logger.info("배치 시작: batch=%s 등록=%d건", batch_id, enqueued)
    return Response({
        "batch_id": str(batch.id),
        "enqueued": enqueued,
        "total": batch.total,
        "error": batch.error_count,
    })


@api_view(["GET"])
def batch_result(request, batch_id):
    """GET /api/batches/{batch_id} — 요약(총/완료/에러) + Job 리스트."""
    batch = get_object_or_404(Batch, pk=batch_id)
    return Response(BatchSerializer(batch).data)


# 워커는 status를 한글 라벨("sTIL 완료", "완료")로 보내고,
# 프론트/내부에서는 영문 키("stil_done", "done")를 쓴다. 둘 다 받아들인다.
_STATUS_LOOKUP = {key: key for key, _ in Job.STATUS_CHOICES}
_STATUS_LOOKUP.update({label: key for key, label in Job.STATUS_CHOICES})

# 워커는 cp_value 하나만 보내지만 모델 컬럼은 JSON이라 dict로 감싸 저장한다.
# lower/upper/abstain 등을 함께 보내오면 같이 담는다.
_CP_SUBFIELDS = ("cp_value", "lower", "upper", "abstain", "decision", "sigma", "n_valid")

# 워커가 직접 덮어쓸 수 있는 컬럼
_WRITABLE_FIELDS = {
    "status",
    "stil_value",
    "conformal_prediction_value",
    "gcs_result_path",
    "medgemma_report",
    "error_message",
}


@api_view(["GET", "PATCH"])
def job_detail(request, job_id):
    """GET  /api/jobs/{job_id}   — 프론트가 개별 결과를 조회
    PATCH /api/jobs/{job_id}/   — 워커(담당자 E)가 처리 결과를 써 넣음

    워커 호출 예시:
      1차: {"status": "sTIL 완료", "stil_value": 4.75, "cp_value": 0.88}
      2차: {"status": "완료", "medgemma_report": {...}}
    """
    job = get_object_or_404(Job, pk=job_id)

    if request.method == "GET":
        # 프론트가 <img src>로 바로 쓸 수 있게 서명 URL을 함께 내려준다.
        return Response(JobSerializer(job, context={"with_urls": True}).data)

    # --- 아래는 PATCH (워커의 결과 기록) ---
    updates = {k: v for k, v in request.data.items() if k in _WRITABLE_FIELDS}

    # status가 한글 라벨로 오면 영문 키로 변환한다.
    if "status" in updates:
        raw = updates["status"]
        key = _STATUS_LOOKUP.get(raw)
        if key is None:
            return Response(
                {"detail": f"허용되지 않는 status 값입니다: {raw}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        updates["status"] = key

    # cp_value 등 conformal 관련 값들을 JSON 컬럼 하나로 합쳐 저장한다.
    cp = {k: request.data[k] for k in _CP_SUBFIELDS if k in request.data}
    if cp:
        merged = dict(job.conformal_prediction_value or {})
        merged.update(cp)
        updates["conformal_prediction_value"] = merged

    if not updates:
        return Response(
            {"detail": "갱신할 필드가 없습니다."}, status=status.HTTP_400_BAD_REQUEST
        )

    for field, value in updates.items():
        setattr(job, field, value)
    job.save()

    logger.info("Job 갱신(PATCH): job=%s fields=%s", job_id, list(updates.keys()))
    return Response(JobSerializer(job).data)


@api_view(["POST"])
def callback(request, job_id):
    """POST /api/internal/jobs/{job_id}/callback — 워커가 결과를 써 넣음(Job 단위, 기존과 동일).

    요청 예시: {"status": "done", "stil_value": 0.42, ...}  (status는 키: waiting/processing/error/done)
    """
    job = get_object_or_404(Job, pk=job_id)

    # 갱신 허용 필드만 화이트리스트로 제한
    allowed = {
        "status",
        "stil_value",
        "conformal_prediction_value",
        "gcs_result_path",
        "medgemma_report",
        "error_message",
    }
    updates = {k: v for k, v in request.data.items() if k in allowed}
    if not updates:
        return Response(
            {"detail": "갱신할 필드가 없습니다."}, status=status.HTTP_400_BAD_REQUEST
        )

    # status가 넘어왔다면 정의된 값인지 확인
    new_status = updates.get("status")
    if new_status and new_status not in dict(Job.STATUS_CHOICES):
        return Response(
            {"detail": f"허용되지 않는 status 값입니다: {new_status}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    for field, value in updates.items():
        setattr(job, field, value)
    job.save()

    logger.info("Job 갱신: job=%s fields=%s", job_id, list(updates.keys()))
    return Response(JobSerializer(job).data)