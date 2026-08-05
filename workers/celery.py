from celery import Celery

app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=[
        "workers.image",
        "workers.audio",
        "workers.video",
        "workers.document"
    ]
)

