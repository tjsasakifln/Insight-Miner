from celery import Celery
from loguru import logger
import redis.asyncio as redis
import asyncio
from .utils import send_email
from .database import SessionLocal, UploadHistory # Import necessary models

CELERY_BROKER_URL = "redis://redis:6379/0"
CELERY_RESULT_BACKEND = "redis://redis:6379/0"
REDIS_URL = "redis://redis:6379/0"

celery_app = Celery(
    "insight_miner_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

@celery_app.task(name="process_file_task")
def process_file_task(file_path: str, upload_id: int, user_email: str):
    logger.info(f"Processing file: {file_path} for upload ID: {upload_id}")
    r = redis.from_url(REDIS_URL)
    
    # Update status in DB and notify via WebSocket
    db = SessionLocal()
    try:
        upload_entry = db.query(UploadHistory).filter(UploadHistory.id == upload_id).first()
        if upload_entry:
            upload_entry.status = "in_progress"
            db.commit()
            asyncio.run(r.publish("progress_updates", f"Upload {upload_id}: Processing started."))

        # Simulate file processing
        for i in range(1, 6):
            time.sleep(2) # Simulate a long-running task
            progress_message = f"Upload {upload_id}: {i*20}% complete."
            logger.info(progress_message)
            asyncio.run(r.publish("progress_updates", progress_message))

        # Update status in DB and notify via WebSocket
        if upload_entry:
            upload_entry.status = "completed"
            db.commit()
            asyncio.run(r.publish("progress_updates", f"Upload {upload_id}: Processing completed."))
            send_email(user_email, "Insight Miner: Análise Concluída", f"Sua análise para o arquivo {upload_entry.file_name} foi concluída.")

        logger.info(f"Finished processing file: {file_path} for upload ID: {upload_id}")
        return {"status": "completed", "file_path": file_path, "upload_id": upload_id}
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        if upload_entry:
            upload_entry.status = "failed"
            db.commit()
            asyncio.run(r.publish("progress_updates", f"Upload {upload_id}: Processing failed."))
        send_email(user_email, "Insight Miner: Análise Falhou", f"Sua análise para o arquivo {upload_entry.file_name} falhou. Erro: {e}")
        raise
    finally:
        db.close()