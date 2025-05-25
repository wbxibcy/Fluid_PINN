#!/bin/bash
# 启动 Celery CPU Worker（只处理 fvm 队列）

echo "Starting CPU worker (FVM)..."

celery -A app.services.tasks worker \
  --loglevel=info \
  --pool=prefork \
  --concurrency=4 \
  -Q fvm \
  --hostname=cpu@%h
