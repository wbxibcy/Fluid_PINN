from celery import Celery

app = Celery('tasks', broker='amqp://xx:xxpassword@localhost//', backend='rpc://')

@app.task
def add(x, y):
    return x + y
