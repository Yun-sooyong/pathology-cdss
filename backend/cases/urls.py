from django.urls import path

from . import views

urlpatterns = [
    path("health", views.health, name="health"),
    path("upload", views.upload, name="upload"),                               # (구) 서버 경유 업로드
    # 대용량 업로드 — 서명 URL 발급 후 프론트가 GCS에 직접 PUT
    path("batches", views.create_batch, name="batch-create"),
    path("batches/<uuid:batch_id>/start", views.start_batch, name="batch-start"),
    path("batches/<uuid:batch_id>", views.batch_result, name="batch-result"),  # 요약 + 리스트
    # 프론트엔드가 쓰는 별칭 — 위 batches와 동일한 응답 (공통 인터페이스 명칭 유지)
    path("result/<uuid:batch_id>", views.batch_result, name="batch-result-alias"),
    path("jobs/<uuid:job_id>", views.job_result, name="job-result"),           # 개별 결과
    path(
        "internal/jobs/<uuid:job_id>/callback",
        views.callback,
        name="job-callback",
    ),
]
