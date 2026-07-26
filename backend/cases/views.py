import logging 

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Batch, Job
from .serializers import BatchCreateSerializer, BatchSerializer, JobSerializer
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