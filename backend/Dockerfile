# 로컬 개발 Python 버전과 일치시킨다 (3.10 미만은 Django 5.2 설치 불가)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# psycopg2-binary는 휠로 설치되므로 빌드 도구 불필요.
# libpq는 런타임에 필요할 수 있어 최소 패키지만 설치한다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 의존성 레이어를 먼저 캐시
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# 정적 파일 수집 (admin 등) — DB 접속 없이 실행 가능해야 함
RUN python manage.py collectstatic --noinput

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
