
import os
from pathlib import Path

from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

def env_bool(key: str, default: str = "False") -> bool:
    return os.environ.get(key, default).lower() in ("true", "1", "yes")


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY')

# 배포에서는 반드시 False여야 한다. 로컬은 .env의 DJANGO_DEBUG=True로 켠다.
# (기본값을 False로 둬서, 환경변수를 깜빡해도 운영에서 켜지지 않도록 안전하게)
DEBUG = env_bool("DJANGO_DEBUG", "False")

ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if h.strip()
]

# Cloud Run은 HTTPS 프록시 뒤에 있으므로 CSRF origin을 명시해야 한다.
# 예) CSRF_TRUSTED_ORIGINS=https://bcss-api-xxxx.run.app
CSRF_TRUSTED_ORIGINS = [
    o.strip()
    for o in os.environ.get("CSRF_TRUSTED_ORIGINS", "").split(",")
    if o.strip()
]

# Cloud Run은 TLS를 프록시(로드밸런서)에서 끊고 내부로는 HTTP로 전달한다.
# 이 헤더가 있어야 Django가 "원래는 HTTPS 요청"임을 인식해 CSRF/보안쿠키를 올바로 처리한다.
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# 운영(DEBUG=False)일 때만 켜는 보안 설정 — 로컬 개발엔 영향 없음
if not DEBUG:
    SESSION_COOKIE_SECURE = True   # 세션 쿠키를 HTTPS로만 전송
    CSRF_COOKIE_SECURE = True      # CSRF 쿠키를 HTTPS로만 전송

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # 서드 파티 
    'rest_framework',
    'corsheaders',
    # 내앱
    'cases',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',   # 정적 파일 서빙 (SecurityMiddleware 바로 다음)
    'corsheaders.middleware.CorsMiddleware',         # CORS 처리 (CommonMiddleware보다 위에 위치)
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# 로컬 테스트 플래그: True면 실제 GCS/큐 호출을 생략(gcs.py·tasks.py)하고 DB도 SQLite를 쓴다.
LOCAL_FAKE_GCP = env_bool("LOCAL_FAKE_GCP", "False")
# 배포(Cloud Run) 환경에서만 채워짐 — 값이 있으면 Cloud SQL(Unix 소켓) 분기를 탄다.
INSTANCE_CONNECTION_NAME = os.environ.get('INSTANCE_CONNECTION_NAME')

# PostgreSQL 공통 접속 정보 (로컬/배포 분기에서 공통으로 재사용)
_db_common = {
    "ENGINE": "django.db.backends.postgresql",
    "NAME": os.environ.get("DB_NAME"),
    "USER": os.environ.get("DB_USER"),
    "PASSWORD": os.environ.get("DB_PASSWORD"),
    "CONN_MAX_AGE": 60,
}

if LOCAL_FAKE_GCP:
    # 로컬 공부/테스트: DB 서버 없이 파일 DB(SQLite) 사용
    DATABASES = {
        "default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}
    }
elif INSTANCE_CONNECTION_NAME:
    # Cloud Run 배포: Cloud SQL에 Unix 소켓으로 접속
    DATABASES = {
        "default": {
            **_db_common,
            "HOST": f"/cloudsql/{INSTANCE_CONNECTION_NAME}",
            "PORT": os.environ.get("DB_PORT"),
        }
    }
else:
    # 로컬에서 진짜 PostgreSQL을 쓸 때 (Cloud SQL Auth Proxy 등)
    DATABASES = {
        "default": {
            **_db_common,
            "HOST": os.environ.get("DB_HOST"),
            "PORT": os.environ.get("DB_PORT"),
        }
    }

# React 프론트엔드에서 오는 요청만 허용.
# 브라우저는 localhost와 127.0.0.1을 다른 origin으로 보므로 둘 다 기본값에 넣는다.
# 배포 시에는 env.yaml의 CORS_ALLOWED_ORIGINS에 실제 프론트 도메인을 추가한다.
CORS_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

# ---- 업로드 ----
# (구) 서버 경유 업로드에서 쓰는 이미지 용량 상한.
# 실측 병리 이미지는 최대 141MB지만, Cloud Run 요청 제한(32MB)이 먼저 걸리므로
# 큰 파일은 서명 URL 방식(POST /api/batches)으로 GCS에 직접 올린다.
MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "200"))

# 서명 업로드 URL 유효시간. 141MB 파일 업로드 시간을 감안해 넉넉히 잡는다.
SIGNED_URL_EXPIRE_MINUTES = int(os.environ.get("SIGNED_URL_EXPIRE_MINUTES", "60"))

# ---- GCP: 스토리지 ----
# 업로드 이미지가 저장될 GCS 버킷과 경로 접두어 (가짜 모드에서도 prefix는 사용)
GCS_DATA_BUCKET = os.environ.get("GCS_DATA_BUCKET")
GCS_UPLOAD_PREFIX = os.environ.get("GCS_UPLOAD_PREFIX", "uploads")
# 워커가 세그멘테이션 결과 이미지를 올릴 위치
GCS_RESULT_PREFIX = os.environ.get("GCS_RESULT_PREFIX", "results")

# ---- GCP: Cloud Tasks (추론 작업 큐) — tasks.py에서 사용 ----
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
CLOUD_TASKS_LOCATION = os.environ.get("CLOUD_TASKS_LOCATION")
CLOUD_TASKS_QUEUE = os.environ.get("CLOUD_TASKS_QUEUE")
# Cloud Tasks가 Cloud Run Admin API를 호출할 때 쓸 서비스 계정.
# 이 SA에 run.jobs.runWithOverrides 권한이 있어야 Job 실행이 가능하다.
CLOUD_TASKS_INVOKER_SA = os.environ.get("CLOUD_TASKS_INVOKER_SA", "")

# ---- 워커 (Cloud Run Job) ----
# 워커는 HTTP 서버가 아닌 Job이라, 이름/리전으로 :run 엔드포인트를 조립해 트리거한다.
WORKER_JOB_NAME = os.environ.get("WORKER_JOB_NAME", "cdss-worker")
WORKER_JOB_REGION = os.environ.get("WORKER_JOB_REGION", "asia-northeast3")

# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR /'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ---- 로깅 ----
# Cloud Run은 컨테이너의 stdout/stderr를 Cloud Logging으로 자동 수집한다.
# 콘솔 핸들러를 명시해, 운영(DEBUG=False)에서도 예외 트레이스백이 로그에 남게 한다.
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "[{levelname}] {name}: {message}", "style": "{"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "simple"},
    },
    # 전체 기본: INFO 이상을 콘솔로
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        # 뷰에서 처리 안 된 예외(500)의 트레이스백을 콘솔에 남긴다 (이번 디버깅 교훈)
        "django.request": {"handlers": ["console"], "level": "ERROR", "propagate": False},
        # 우리 앱(cases)의 logger.info / logger.exception 출력
        "cases": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}
