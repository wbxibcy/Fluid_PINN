#!/bin/bash
# 启动 Celery GPU Worker（只处理 pinn 队列）

echo "Starting GPU worker (PINN)..."

celery -A app.services.tasks worker \
  --loglevel=info \
  --pool=solo \
  -Q pinn \
  --hostname=gpu@%h


