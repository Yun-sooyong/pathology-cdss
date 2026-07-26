"""
GCS 업로드 연결
"""

import logging
import uuid 
from pathlib import PurePosixPath

from django.conf import settings 
from google.cloud import storage 

logger = logging.getLogger(__name__)

# 캐싱 (클라이언트 재사용)
_client = None 

def _get_client() -> storage.Client:
    """storage.Client를 지연 생성해서 재사용한다.

    모듈 import 시점에 만들면 인증 정보가 없는 환경(테스트 등)에서
    import 자체가 실패하므로 첫 호출 때 생성한다.
    """
    global _client
    if _client is None:
        _client = storage.Client()
    return _client 

def upload_image(file_obj, original_filename: str) -> str:
    """업로드된 이미지를 GCS에 저장하고 gs:// 경로를 반환한다.

    Args:
        file_obj: Django의 UploadedFile 객체
        original_filename: 원본 파일명 (확장자 추출용)

    Returns:
        "gs://버킷명/uploads/<uuid>.png" 형식의 경로
    """
    # 로컬 테스트: 실제 업로드 없이 가짜 경로만 돌려준다.
    if settings.LOCAL_FAKE_GCP:
        suffix = PurePosixPath(original_filename).suffix.lower() or ".png"
        return f"gs://fake-bucket/{settings.GCS_UPLOAD_PREFIX}/{uuid.uuid4()}{suffix}"

    # 원본 확장자만 살리고, 파일명은 UUID로 새로 만든다.
    # (사용자가 올린 파일명을 그대로 쓰면 겹치거나 위험한 이름이 들어올 수 있음)
    suffix = PurePosixPath(original_filename).suffix.lower() or ".png"
    blob_name = f"{settings.GCS_UPLOAD_PREFIX}/{uuid.uuid4()}{suffix}"
    
    # 버킷 안의 저장 위치(blob)를 지정
    bucket = _get_client().bucket(settings.GCS_DATA_BUCKET)
    blob = bucket.blob(blob_name)
    
    # 파일 포인터를 맨 앞으로 되돌린 뒤 업로드 
    # 파일을 이미 읽었을 수 있어서 seek(0)이 필요 
    file_obj.seek(0)
    blob.upload_from_file(
        file_obj, 
        # content_type을 못 구하면 일반 바이너리 타입으로 처리 
        content_type=getattr(file_obj, "content_type", None) or "application/octet-stream",
    )
    
    # DB에는 gs:// 경로만 저장, 파일은 gcs 에 
    gcs_path = f"gs://{settings.GCS_DATA_BUCKET}/{blob_name}"
    logger.info("이미지 업로드 완료: %s", gcs_path)
    return gcs_path