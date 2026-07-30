from django.urls import path

from . import views

urlpatterns = [
    path("health", views.health, name="health"),
    path("upload", views.upload, name="upload"),                               # (구) 서버 경유 업로드
    # 대용량 업로드 — 서명 URL 발급 후 프론트가 GCS에 직접 PUT
    # 대시보드용 최근 케이스 목록 (batch_id를 몰라도 조회 가능)
    path("cases", views.case_list, name="case-list"),
    path("batches", views.create_batch, name="batch-create"),
    path("batches/<uuid:batch_id>/start", views.start_batch, name="batch-start"),
    path("batches/<uuid:batch_id>", views.batch_result, name="batch-result"),  # 요약 + 리스트
    # 프론트엔드가 쓰는 별칭 — 위 batches와 동일한 응답 (공통 인터페이스 명칭 유지)
    path("result/<uuid:batch_id>", views.batch_result, name="batch-result-alias"),
    # 개별 결과: GET(프론트 조회) / PATCH(워커 결과 기록)
    # 워커가 f"{BASE}/jobs/{id}/" 처럼 끝 슬래시를 붙여 호출하므로 두 형태 모두 등록한다.
    path("jobs/<uuid:job_id>", views.job_detail, name="job-detail"),
    path("jobs/<uuid:job_id>/", views.job_detail, name="job-detail-slash"),
    path(
        "internal/jobs/<uuid:job_id>/callback",
        views.callback,
        name="job-callback",
    ),
]
