import logging 

from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Batch, Job
from .serializers import (
    BatchCreateSerializer,
    BatchInitSerializer,
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


@api_view(["GET"])
def job_result(request, job_id):
    """GET /api/jobs/{job_id} — 리스트에서 항목 클릭 시 개별 결과."""
    job = get_object_or_404(Job, pk=job_id)
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