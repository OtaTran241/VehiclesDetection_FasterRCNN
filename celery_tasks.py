# File: celery_tasks.py
import os
from celery import Celery
from detection.detection_model import run_detection
import uuid

celery_app = Celery('celery_tasks', 
                    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
                    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'))

@celery_app.task(bind=True)
def detect_object(self, file_path):
    task_id = self.request.id or uuid.uuid4()
    result = run_detection(file_path)
    result['task_id'] = str(task_id)
    return result
