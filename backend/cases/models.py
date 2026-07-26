import uuid
from django.db import models


class Batch(models.Model):
    """한 번의 업로드로 묶인 이미지들의 그룹 (이미지 1~15장 = Batch 1개 + Job 여러 개).
    총/완료/에러 개수는 컬럼으로 저장하지 않고, 소속 Job을 세서 계산한다(항상 정확).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model_name = models.CharField(max_length=50)  # 이 묶음에 사용한 모델(모든 Job 공통)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "batches"
        ordering = ["-created_at"]

    @property
    def total(self):
        return self.jobs.count()                                    # 총 장수

    @property
    def done_count(self):
        return self.jobs.filter(status=Job.STATUS_DONE).count()     # 완료 개수

    @property
    def error_count(self):
        return self.jobs.filter(status=Job.STATUS_ERROR).count()    # 에러 개수

    @property
    def status(self):
        # 아직 안 끝난(대기/처리중) Job이 하나라도 있으면 '처리중', 다 끝났으면 '완료'
        unfinished = self.jobs.exclude(
            status__in=[Job.STATUS_DONE, Job.STATUS_ERROR]
        ).exists()
        return Job.STATUS_PROCESSING if unfinished else Job.STATUS_DONE

    def __str__(self):
        return f"Batch {self.id} ({self.model_name})"


class Job(models.Model) :
    # 정의된 Job table 구현 

    # 작업 상태 값 정의 
    STATUS_WAITING = "waiting"       # queue 등록 직후, 시작전 
    STATUS_PROCESSING = "processing" # 실행 중, 추론 진행 
    STATUS_ERROR = "error"           # 오류 발생 
    STATUS_DONE = "done"             # 추론 완료, 결과필드 생성 
    
    # (키, 라벨) 형태
    # choice 에 넣으면 admin/serializer 가 허용된 키만 받도록 검증 가능
    # get_status_display() 로 한글 라벨을 꺼내서 쓸 수 있음 
    STATUS_CHOICES = [
        (STATUS_WAITING, "대기"),
        (STATUS_PROCESSING, "진행중"),
        (STATUS_ERROR, "오류"),
        (STATUS_DONE, "완료"),
    ]
    
    # 추론 모델 종류 
    MODEL_UNET_V1 = "unet_v1"
    MODEL_ATTENTION_UNET_V1 = "attention_unet_v1"
    
    MODEL_CHOICES = [
        (MODEL_UNET_V1, "unet_v1"),
        (MODEL_ATTENTION_UNET_V1, "attention_unet_v1"),
    ]
    
    ## 실제 DB 필드 ##
    
    # URL의 숫자로 내용 추측을 못하게 하기 위해 uuid 를 기본키로 사용
    id = models.UUIDField(
        primary_key=True,   # 숫자 대신 uuid 사용
        default=uuid.uuid4, # 새로운 데이터 생성마다 무작위 uuid 생성
        editable=False,     # 어드민 페이지나 API 입력 폼에서 이 필드가 수정되지 않도록 보호
    )
    
    # 어느 묶음(Batch) 소속인지. 묶음 삭제 시 소속 Job도 함께 삭제(CASCADE).
    # null 허용은 기존 데이터 호환용 — 새로 만드는 Job은 항상 batch가 채워진다.
    batch = models.ForeignKey(
        Batch, related_name="jobs", on_delete=models.CASCADE, null=True, blank=True
    )

    # 원본이미지 경로
    gcs_image_path = models.CharField(max_length=500)
    # 모델 이름 
    # choice 를 사용해서 무작위 값 입력을 막고 choice 내부에서 선택하도록함 
    model_name = models.CharField(max_length=50,choices=MODEL_CHOICES)
    # 상태는 조회/필터에 사용되므로 인덱스를 걸어 빠른 검색이 되도록 함 
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_WAITING,
        db_index=True,
    )
    # 추론 결과 필드 (값이 나오기 전에는 비어있으므로 null 허용)
    stil_value  = models.FloatField(null=True, blank=True) # sTIL 예측값
    conformal_prediction_value = models.JSONField(null=True, blank=True) # 신뢰구간등 수치 JSON
    gcs_result_path = models.CharField(max_length=500, null=True, blank=True) # 결과 이미지 경로 
    medgemma_report = models.JSONField(null=True, blank=True) # VLM 리포트
    
    # 오류 기록 
    error_message = models.TextField(null=True, blank=True)

    # 생성/수정 시간 기록 
    created_at = models.DateTimeField(auto_now_add=True) # row 최초 생성시 저장
    updated_at = models.DateTimeField(auto_now = True) # save 한 시간 저장
    
    class Meta: 
        db_table = "jobs" # 테이블 이름 고정
        ordering = ["-created_at"] # 조회 시 기본적으로 최신순 정렬 
        
    def __str__(self) :
        # admin 목록이나 로그에 객체가 찍힐 때 깔끔하게 표시
        return f"{self.id} ({self.model_name} / {self.status})"
    
    