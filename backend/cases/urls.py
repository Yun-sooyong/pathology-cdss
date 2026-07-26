from django.urls import path

from . import views

urlpatterns = [
    path("health", views.health, name="health"),
    path("upload", views.upload, name="upload"),                               # 1~15장 업로드
    path("batches/<uuid:batch_id>", views.batch_result, name="batch-result"),  # 요약 + 리스트
    path("jobs/<uuid:job_id>", views.job_result, name="job-result"),           # 개별 결과
    path(
        "internal/jobs/<uuid:job_id>/callback",
        views.callback,
        name="job-callback",
    ),
]
