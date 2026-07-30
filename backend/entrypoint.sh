#!/bin/sh
set -e

echo "[entrypoint] 마이그레이션 적용 중..."
python manage.py migrate --noinput

echo "[entrypoint] gunicorn 기동 (port=${PORT})"
exec gunicorn config.wsgi:application \
    --bind "0.0.0.0:${PORT}" \
    --workers 2 \
    --threads 4 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
