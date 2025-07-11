from celery import Celery
from loguru import logger
import redis.asyncio as redis
import asyncio
import pandas as pd
import io

from .utils import send_email
from .database import SessionLocal, UploadHistory, AnalysisMetadata # Import necessary models
from .sentiment_analysis import sentiment_analyzer

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
    db = SessionLocal()
    upload_entry = None
    try:
        upload_entry = db.query(UploadHistory).filter(UploadHistory.id == upload_id).first()
        if upload_entry:
            upload_entry.status = "in_progress"
            db.commit()
            asyncio.run(r.publish("progress_updates", f"Upload {upload_id}: Processing started."))

        # Read file in chunks
        chunk_size = 1000
        df_chunks = pd.read_csv(file_path, chunksize=chunk_size)
        total_reviews = 0
        processed_reviews = 0

        for i, chunk in enumerate(df_chunks):
            total_reviews += len(chunk)
            logger.info(f"Processing chunk {i+1} with {len(chunk)} reviews.")
            
            # Dispatch sentiment analysis for each review in the chunk
            for index, row in chunk.iterrows():
                review_text = row["review_text"] # Assuming 'review_text' column
                # Asynchronous call to sentiment analysis, but Celery task is synchronous
                # For true async, this would be another Celery task or a direct async call if not blocking
                sentiment_result = asyncio.run(sentiment_analyzer.analyze_sentiment(review_text))
                
                # Store analysis metadata (simplified for now)
                analysis_metadata = AnalysisMetadata(
                    upload_id=upload_id,
                    analysis_type="sentiment",
                    status="completed",
                    result_summary=str(sentiment_result), # Store as string for simplicity
                    analyst_id=upload_entry.uploader_id # Assuming uploader is the analyst
                )
                db.add(analysis_metadata)
                db.commit()
                db.refresh(analysis_metadata)

                processed_reviews += 1
                progress_message = f"Upload {upload_id}: Processed {processed_reviews}/{total_reviews} reviews."
                asyncio.run(r.publish("progress_updates", progress_message))

        # Update status in DB and notify via WebSocket
        if upload_entry:
            upload_entry.status = "completed"
            db.commit()
            asyncio.run(r.publish("progress_updates", f"Upload {upload_id}: Processing completed. Total reviews: {total_reviews}"))
            send_email(user_email, "Insight Miner: Análise Concluída", f"Sua análise para o arquivo {upload_entry.file_name} foi concluída. Total de reviews processados: {total_reviews}")

        logger.info(f"Finished processing file: {file_path} for upload ID: {upload_id}. Total reviews: {total_reviews}")
        return {"status": "completed", "file_path": file_path, "upload_id": upload_id, "total_reviews": total_reviews}
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
