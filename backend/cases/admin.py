from django.contrib import admin
from .models import Job, Batch


# 묶음(Batch)을 admin 페이지에서 조회 — 요약 개수를 한눈에 보기 위함
@admin.register(Batch)
class BatchAdmin(admin.ModelAdmin):
    # total/done_count/error_count/status는 모델의 @property라 목록에 그대로 쓸 수 있다.
    list_display = ("id", "model_name", "status", "total", "done_count", "error_count", "created_at")
    readonly_fields = ("id", "created_at", "updated_at")


# 개별 작업(Job)을 admin 페이지에서 조회/필터
@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    # 목록에 노출한 columns
    list_display = ("id", "model_name", "status", "stil_value", "created_at") 
    list_filter = ("status", "model_name") # 우측 필터 (상태, 모델별)
    search_fields = ("id",) # 상단 검색창
    readonly_fields = ("id", "created_at", "updated_at") # 수정불가