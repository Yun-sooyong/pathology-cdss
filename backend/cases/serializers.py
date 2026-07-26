from django.conf import settings
from rest_framework import serializers

from .models import Job


class BatchCreateSerializer(serializers.Serializer):
    """POST /upload 요청 검증 — 이미지 1~15장 + model_name (요구사항 1)."""

    images = serializers.ListField(
        child=serializers.ImageField(),  # 리스트의 각 원소가 이미지인지 검증
        min_length=1,
        max_length=15,                    # 한 번에 1~15장으로 제한
    )
    model_name = serializers.ChoiceField(choices=Job.MODEL_CHOICES)

    def validate_images(self, value):
        # 이미지마다 용량 상한을 개별 확인
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        for img in value:
            if img.size > max_bytes:
                raise serializers.ValidationError(
                    f"'{img.name}'의 크기가 {settings.MAX_UPLOAD_SIZE_MB}MB를 초과합니다."
                )
        return value


class UploadFileSerializer(serializers.Serializer):
    """서명 URL을 발급받을 파일 하나의 정보."""

    filename = serializers.CharField(max_length=255)
    # 브라우저가 PUT할 때 보낼 Content-Type. 서명에 포함되므로 값이 일치해야 한다.
    content_type = serializers.CharField(max_length=100, default="application/octet-stream")


class BatchInitSerializer(serializers.Serializer):
    """POST /api/batches 요청 검증 — 파일은 받지 않고 목록만 받는다.

    실제 파일은 발급된 서명 URL로 GCS에 직접 업로드된다(Cloud Run 32MB 제한 회피).
    """

    model_name = serializers.ChoiceField(choices=Job.MODEL_CHOICES)
    files = serializers.ListField(
        child=UploadFileSerializer(),
        min_length=1,
        max_length=15,  # 한 배치당 1~15장
    )


class JobListItemSerializer(serializers.ModelSerializer):
    """Batch 안의 '리스트 항목'용 — 목록 표시에 필요한 최소 정보만 (요구사항 3)."""

    job_id = serializers.UUIDField(source="id", read_only=True)
    status_label = serializers.CharField(source="get_status_display", read_only=True)  # 한글 라벨

    class Meta:
        model = Job
        fields = ["job_id", "status", "status_label", "gcs_image_path"]


class BatchSerializer(serializers.Serializer):
    """GET /batches/{batch_id} 응답 — 요약(총/완료/에러) + Job 리스트 (요구사항 2·3)."""

    batch_id = serializers.UUIDField(source="id", read_only=True)
    model_name = serializers.CharField(read_only=True)
    status = serializers.CharField(read_only=True)                     # 전체 진행 상태
    total = serializers.IntegerField(read_only=True)                   # 총 장수
    done = serializers.IntegerField(source="done_count", read_only=True)   # 완료 개수
    error = serializers.IntegerField(source="error_count", read_only=True) # 에러 개수
    jobs = JobListItemSerializer(many=True, read_only=True)            # 리스트
    created_at = serializers.DateTimeField(read_only=True)


class JobSerializer(serializers.ModelSerializer):
    """GET /jobs/{job_id} 응답 — 항목 클릭 시 보여줄 개별 결과 전체 (요구사항 3)."""

    job_id = serializers.UUIDField(source="id", read_only=True)

    class Meta:
        model = Job
        fields = [
            "job_id",
            "status",
            "model_name",
            "gcs_image_path",
            "stil_value",
            "conformal_prediction_value",
            "gcs_result_path",
            "medgemma_report",
            "error_message",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields