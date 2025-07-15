"""
Enterprise-grade Celery worker with proper async handling and performance optimizations.
"""
import json
import time
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import threading
import concurrent.futures
from functools import wraps

from celery import Celery, Task
from celery.signals import worker_init, worker_shutdown
from celery.exceptions import Retry
import pandas as pd
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .config import settings
from .logger import get_logger
from .database import UploadHistory, AnalysisMetadata, engine
from .sentiment_analysis import sentiment_analyzer
from .utils import send_email

logger = get_logger(__name__)

# Configuration
CELERY_CONFIG = {
    'broker_url': settings.redis.connection_string,
    'result_backend': settings.redis.connection_string,
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
    'task_track_started': True,
    'task_time_limit': 3600,  # 1 hour max
    'task_soft_time_limit': 3000,  # 50 minutes soft limit
    'worker_prefetch_multiplier': 1,
    'task_acks_late': True,
    'worker_max_tasks_per_child': 1000,
    'task_routes': {
        'insight_miner.tasks.process_file_task': {'queue': 'file_processing'},
        'insight_miner.tasks.batch_sentiment_analysis': {'queue': 'sentiment_analysis'},
        'insight_miner.tasks.health_check': {'queue': 'health'},
    },
    'task_default_queue': 'default',
    'task_default_exchange': 'default',
    'task_default_routing_key': 'default',
}

# Create Celery app
celery_app = Celery("insight_miner_tasks")
celery_app.config_from_object(CELERY_CONFIG)

# Global resources
_redis_pool = None
_session_factory = None
_thread_pool = None


@dataclass
class ProcessingResult:
    """Result structure for processing operations."""
    status: str
    file_path: str
    upload_id: int
    total_reviews: int
    processed_reviews: int
    failed_reviews: int
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class ProgressUpdate:
    """Progress update structure."""
    upload_id: int
    status: str
    progress_percent: float
    current_item: int
    total_items: int
    message: str
    timestamp: datetime


class AsyncBridge:
    """Bridge between sync Celery and async operations."""
    
    def __init__(self):
        self.loop = None
        self.thread = None
        self._setup_async_bridge()
    
    def _setup_async_bridge(self):
        """Setup async event loop in separate thread."""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        
        # Wait for loop to be ready
        while self.loop is None:
            time.sleep(0.01)
    
    def run_async(self, coro):
        """Run async coroutine in the bridge thread."""
        if self.loop is None:
            raise RuntimeError("Async bridge not initialized")
        
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=30)
    
    def shutdown(self):
        """Shutdown async bridge."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5)


# Global async bridge
async_bridge = AsyncBridge()


class EnhancedTask(Task):
    """Enhanced Celery task with monitoring and error handling."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Task success handler."""
        logger.info(f"Task {task_id} completed successfully", 
                   task_id=task_id, result=retval)
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Task failure handler."""
        logger.error(f"Task {task_id} failed", 
                    task_id=task_id, 
                    exception=str(exc),
                    traceback=einfo.traceback)
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Task retry handler."""
        logger.warning(f"Task {task_id} retrying", 
                      task_id=task_id, 
                      exception=str(exc))


@worker_init.connect
def init_worker(**kwargs):
    """Initialize worker resources."""
    global _redis_pool, _session_factory, _thread_pool
    
    logger.info("Initializing Celery worker")
    
    # Initialize Redis connection pool
    _redis_pool = redis.ConnectionPool.from_url(
        settings.redis.connection_string,
        max_connections=settings.redis.max_connections
    )
    
    # Initialize database session factory
    _session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False
    )
    
    # Initialize thread pool for CPU-bound tasks
    _thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=settings.application.max_concurrent_tasks
    )
    
    logger.info("Celery worker initialized successfully")


@worker_shutdown.connect
def shutdown_worker(**kwargs):
    """Cleanup worker resources."""
    logger.info("Shutting down Celery worker")
    
    if _thread_pool:
        _thread_pool.shutdown(wait=True)
    
    if _redis_pool:
        _redis_pool.disconnect()
    
    async_bridge.shutdown()
    
    logger.info("Celery worker shutdown completed")


@contextmanager
def get_db_session():
    """Get database session with proper cleanup."""
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_redis_client():
    """Get Redis client."""
    return redis.Redis(connection_pool=_redis_pool)


async def publish_progress(upload_id: int, progress: ProgressUpdate):
    """Publish progress update to WebSocket clients."""
    r = redis.Redis(connection_pool=_redis_pool)
    
    progress_data = {
        **asdict(progress),
        'timestamp': progress.timestamp.isoformat()
    }
    
    await r.publish("progress_updates", json.dumps(progress_data))


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
            
            return None
        return wrapper
    return decorator


@celery_app.task(base=EnhancedTask, bind=True, name="process_file_task")
def process_file_task(self, file_path: str, upload_id: int, user_email: str) -> Dict[str, Any]:
    """
    Process uploaded file with sentiment analysis.
    
    Args:
        file_path: Path to the uploaded file
        upload_id: Upload record ID
        user_email: User email for notifications
        
    Returns:
        Processing result dictionary
    """
    start_time = time.time()
    
    logger.info(f"Starting file processing", 
               file_path=file_path, 
               upload_id=upload_id)
    
    try:
        # Update upload status
        with get_db_session() as db:
            upload_entry = db.query(UploadHistory).filter(
                UploadHistory.id == upload_id
            ).first()
            
            if not upload_entry:
                raise ValueError(f"Upload record {upload_id} not found")
            
            upload_entry.status = "in_progress"
            db.commit()
        
        # Publish initial progress
        progress = ProgressUpdate(
            upload_id=upload_id,
            status="started",
            progress_percent=0.0,
            current_item=0,
            total_items=0,
            message="Starting file processing",
            timestamp=datetime.utcnow()
        )
        
        async_bridge.run_async(publish_progress(upload_id, progress))
        
        # Process file in chunks
        result = _process_file_chunks(file_path, upload_id, user_email)
        
        # Update final status
        with get_db_session() as db:
            upload_entry = db.query(UploadHistory).filter(
                UploadHistory.id == upload_id
            ).first()
            
            if upload_entry:
                upload_entry.status = "completed"
                db.commit()
        
        # Send completion notification
        send_email(
            user_email,
            "Insight Miner: Análise Concluída",
            f"Sua análise foi concluída. Total de reviews processados: {result.processed_reviews}"
        )
        
        # Publish final progress
        final_progress = ProgressUpdate(
            upload_id=upload_id,
            status="completed",
            progress_percent=100.0,
            current_item=result.processed_reviews,
            total_items=result.total_reviews,
            message="Processing completed successfully",
            timestamp=datetime.utcnow()
        )
        
        async_bridge.run_async(publish_progress(upload_id, final_progress))
        
        logger.info(f"File processing completed successfully", 
                   upload_id=upload_id, 
                   processing_time=time.time() - start_time)
        
        return asdict(result)
        
    except Exception as e:
        logger.error(f"File processing failed", 
                    upload_id=upload_id, 
                    error=str(e))
        
        # Update error status
        with get_db_session() as db:
            upload_entry = db.query(UploadHistory).filter(
                UploadHistory.id == upload_id
            ).first()
            
            if upload_entry:
                upload_entry.status = "failed"
                db.commit()
        
        # Send error notification
        send_email(
            user_email,
            "Insight Miner: Análise Falhou",
            f"Sua análise falhou. Erro: {str(e)}"
        )
        
        # Publish error progress
        error_progress = ProgressUpdate(
            upload_id=upload_id,
            status="failed",
            progress_percent=0.0,
            current_item=0,
            total_items=0,
            message=f"Processing failed: {str(e)}",
            timestamp=datetime.utcnow()
        )
        
        async_bridge.run_async(publish_progress(upload_id, error_progress))
        
        raise


def _process_file_chunks(file_path: str, upload_id: int, user_email: str) -> ProcessingResult:
    """Process file in chunks for better memory management."""
    chunk_size = settings.application.batch_size
    total_reviews = 0
    processed_reviews = 0
    failed_reviews = 0
    
    # First pass: count total reviews
    try:
        total_reviews = sum(1 for _ in pd.read_csv(file_path, chunksize=chunk_size))
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise
    
    # Second pass: process chunks
    chunk_number = 0
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk_number += 1
        
        logger.info(f"Processing chunk {chunk_number} with {len(chunk)} reviews")
        
        # Validate required columns
        required_columns = ['review_text']
        missing_columns = [col for col in required_columns if col not in chunk.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Process chunk in batches
        batch_results = _process_chunk_batch(chunk, upload_id)
        
        # Update counters
        processed_reviews += batch_results['processed']
        failed_reviews += batch_results['failed']
        
        # Update progress
        progress_percent = (processed_reviews / total_reviews) * 100
        
        progress = ProgressUpdate(
            upload_id=upload_id,
            status="processing",
            progress_percent=progress_percent,
            current_item=processed_reviews,
            total_items=total_reviews,
            message=f"Processed {processed_reviews}/{total_reviews} reviews",
            timestamp=datetime.utcnow()
        )
        
        async_bridge.run_async(publish_progress(upload_id, progress))
    
    return ProcessingResult(
        status="completed",
        file_path=file_path,
        upload_id=upload_id,
        total_reviews=total_reviews,
        processed_reviews=processed_reviews,
        failed_reviews=failed_reviews,
        processing_time=time.time()
    )


def _process_chunk_batch(chunk: pd.DataFrame, upload_id: int) -> Dict[str, int]:
    """Process a chunk of reviews with batch sentiment analysis."""
    processed = 0
    failed = 0
    
    # Extract texts for batch processing
    texts = chunk['review_text'].fillna('').astype(str).tolist()
    
    # Batch sentiment analysis
    try:
        sentiment_results = async_bridge.run_async(
            sentiment_analyzer.analyze_batch(texts)
        )
        
        # Store results in database
        with get_db_session() as db:
            upload_entry = db.query(UploadHistory).filter(
                UploadHistory.id == upload_id
            ).first()
            
            if not upload_entry:
                raise ValueError(f"Upload record {upload_id} not found")
            
            # Batch insert analysis metadata
            analysis_records = []
            
            for i, (text, sentiment_result) in enumerate(zip(texts, sentiment_results)):
                if isinstance(sentiment_result, dict) and 'error' not in sentiment_result:
                    analysis_metadata = AnalysisMetadata(
                        upload_id=upload_id,
                        analysis_type="sentiment",
                        status="completed",
                        result_summary=json.dumps(sentiment_result),
                        analyst_id=upload_entry.uploader_id
                    )
                    analysis_records.append(analysis_metadata)
                    processed += 1
                else:
                    logger.warning(f"Sentiment analysis failed for text {i}: {sentiment_result}")
                    failed += 1
            
            # Batch insert for better performance
            if analysis_records:
                db.bulk_save_objects(analysis_records)
                db.commit()
    
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        failed += len(texts)
    
    return {'processed': processed, 'failed': failed}


@celery_app.task(base=EnhancedTask, name="health_check")
def health_check() -> Dict[str, Any]:
    """Celery worker health check."""
    logger.info("Performing health check")
    
    try:
        # Test database connection
        with get_db_session() as db:
            db.execute("SELECT 1")
        
        # Test Redis connection
        r = get_redis_client()
        r.ping()
        
        # Test sentiment analyzer
        test_result = async_bridge.run_async(
            sentiment_analyzer.analyze_sentiment("test message")
        )
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
            "redis": "connected",
            "sentiment_analyzer": "working",
            "test_result": test_result
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
