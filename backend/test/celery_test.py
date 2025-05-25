from tasks_test import add

# 调用 Celery 任务
result = add.apply_async((4, 6))

# 获取任务结果
print("Task result:", result.get())
