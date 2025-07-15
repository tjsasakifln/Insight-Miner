"""
Repository interfaces (ports) for domain entities following Clean Architecture.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from .entities import User, FileUpload, SentimentAnalysis, AnalysisTask
from .value_objects import UserId, Email, UploadId, TaskId, ProcessingStatus, AnalysisType


class Repository(ABC):
    """Base repository interface."""
    
    @abstractmethod
    async def save(self, entity) -> None:
        """Save entity to persistent storage."""
        pass
    
    @abstractmethod
    async def delete(self, entity) -> None:
        """Delete entity from persistent storage."""
        pass
    
    @abstractmethod
    async def find_by_id(self, entity_id: str):
        """Find entity by ID."""
        pass


class UserRepository(Repository):
    """Repository interface for User aggregate."""
    
    @abstractmethod
    async def save(self, user: User) -> None:
        """Save user to persistent storage."""
        pass
    
    @abstractmethod
    async def delete(self, user: User) -> None:
        """Delete user from persistent storage."""
        pass
    
    @abstractmethod
    async def find_by_id(self, user_id: UserId) -> Optional[User]:
        """Find user by ID."""
        pass
    
    @abstractmethod
    async def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email address."""
        pass
    
    @abstractmethod
    async def find_active_users(self) -> List[User]:
        """Find all active users."""
        pass
    
    @abstractmethod
    async def find_users_by_role(self, role: str) -> List[User]:
        """Find users by role."""
        pass
    
    @abstractmethod
    async def find_users_with_failed_logins(self, threshold: int) -> List[User]:
        """Find users with failed login attempts above threshold."""
        pass
    
    @abstractmethod
    async def count_users_by_role(self) -> Dict[str, int]:
        """Count users by role."""
        pass
    
    @abstractmethod
    async def exists_by_email(self, email: Email) -> bool:
        """Check if user exists by email."""
        pass
    
    @abstractmethod
    async def find_users_created_after(self, date: datetime) -> List[User]:
        """Find users created after specific date."""
        pass
    
    @abstractmethod
    async def find_users_with_two_factor_enabled(self) -> List[User]:
        """Find users with two-factor authentication enabled."""
        pass


class FileUploadRepository(Repository):
    """Repository interface for FileUpload aggregate."""
    
    @abstractmethod
    async def save(self, file_upload: FileUpload) -> None:
        """Save file upload to persistent storage."""
        pass
    
    @abstractmethod
    async def delete(self, file_upload: FileUpload) -> None:
        """Delete file upload from persistent storage."""
        pass
    
    @abstractmethod
    async def find_by_id(self, upload_id: UploadId) -> Optional[FileUpload]:
        """Find file upload by ID."""
        pass
    
    @abstractmethod
    async def find_by_uploader_id(self, uploader_id: UserId) -> List[FileUpload]:
        """Find file uploads by uploader ID."""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: ProcessingStatus) -> List[FileUpload]:
        """Find file uploads by processing status."""
        pass
    
    @abstractmethod
    async def find_pending_uploads(self) -> List[FileUpload]:
        """Find all pending file uploads."""
        pass
    
    @abstractmethod
    async def find_in_progress_uploads(self) -> List[FileUpload]:
        """Find all in-progress file uploads."""
        pass
    
    @abstractmethod
    async def find_completed_uploads(self, limit: int = 100) -> List[FileUpload]:
        """Find recently completed file uploads."""
        pass
    
    @abstractmethod
    async def find_failed_uploads(self, limit: int = 100) -> List[FileUpload]:
        """Find failed file uploads."""
        pass
    
    @abstractmethod
    async def find_uploads_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[FileUpload]:
        """Find file uploads within date range."""
        pass
    
    @abstractmethod
    async def count_uploads_by_status(self) -> Dict[ProcessingStatus, int]:
        """Count file uploads by status."""
        pass
    
    @abstractmethod
    async def count_uploads_by_user(self, user_id: UserId) -> int:
        """Count file uploads by user."""
        pass
    
    @abstractmethod
    async def find_large_uploads(self, size_threshold: int) -> List[FileUpload]:
        """Find uploads above size threshold."""
        pass
    
    @abstractmethod
    async def find_uploads_with_errors(self) -> List[FileUpload]:
        """Find uploads with error messages."""
        pass
    
    @abstractmethod
    async def get_upload_statistics(self) -> Dict[str, Any]:
        """Get comprehensive upload statistics."""
        pass


class SentimentAnalysisRepository(Repository):
    """Repository interface for SentimentAnalysis aggregate."""
    
    @abstractmethod
    async def save(self, analysis: SentimentAnalysis) -> None:
        """Save sentiment analysis to persistent storage."""
        pass
    
    @abstractmethod
    async def save_batch(self, analyses: List[SentimentAnalysis]) -> None:
        """Save multiple sentiment analyses in batch."""
        pass
    
    @abstractmethod
    async def delete(self, analysis: SentimentAnalysis) -> None:
        """Delete sentiment analysis from persistent storage."""
        pass
    
    @abstractmethod
    async def find_by_id(self, analysis_id: str) -> Optional[SentimentAnalysis]:
        """Find sentiment analysis by ID."""
        pass
    
    @abstractmethod
    async def find_by_upload_id(self, upload_id: UploadId) -> List[SentimentAnalysis]:
        """Find sentiment analyses by upload ID."""
        pass
    
    @abstractmethod
    async def find_by_review_id(self, review_id: str) -> Optional[SentimentAnalysis]:
        """Find sentiment analysis by review ID."""
        pass
    
    @abstractmethod
    async def find_by_sentiment_range(
        self, 
        min_score: float, 
        max_score: float
    ) -> List[SentimentAnalysis]:
        """Find sentiment analyses within score range."""
        pass
    
    @abstractmethod
    async def find_positive_sentiments(self, upload_id: UploadId) -> List[SentimentAnalysis]:
        """Find positive sentiment analyses for upload."""
        pass
    
    @abstractmethod
    async def find_negative_sentiments(self, upload_id: UploadId) -> List[SentimentAnalysis]:
        """Find negative sentiment analyses for upload."""
        pass
    
    @abstractmethod
    async def find_by_confidence_threshold(
        self, 
        threshold: float
    ) -> List[SentimentAnalysis]:
        """Find analyses above confidence threshold."""
        pass
    
    @abstractmethod
    async def find_by_provider(self, provider: str) -> List[SentimentAnalysis]:
        """Find sentiment analyses by provider."""
        pass
    
    @abstractmethod
    async def find_by_analysis_type(self, analysis_type: AnalysisType) -> List[SentimentAnalysis]:
        """Find sentiment analyses by type."""
        pass
    
    @abstractmethod
    async def find_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[SentimentAnalysis]:
        """Find sentiment analyses within date range."""
        pass
    
    @abstractmethod
    async def count_by_sentiment_category(self, upload_id: UploadId) -> Dict[str, int]:
        """Count analyses by sentiment category for upload."""
        pass
    
    @abstractmethod
    async def calculate_average_sentiment(self, upload_id: UploadId) -> Optional[float]:
        """Calculate average sentiment score for upload."""
        pass
    
    @abstractmethod
    async def find_top_sentiment_scores(
        self, 
        limit: int = 10, 
        positive: bool = True
    ) -> List[SentimentAnalysis]:
        """Find top sentiment scores (positive or negative)."""
        pass
    
    @abstractmethod
    async def find_anomalous_sentiments(
        self, 
        upload_id: UploadId, 
        z_score_threshold: float = 2.0
    ) -> List[SentimentAnalysis]:
        """Find sentiment analyses that are statistical anomalies."""
        pass
    
    @abstractmethod
    async def get_sentiment_statistics(self, upload_id: UploadId) -> Dict[str, Any]:
        """Get comprehensive sentiment statistics for upload."""
        pass
    
    @abstractmethod
    async def find_slow_analyses(self, time_threshold: float) -> List[SentimentAnalysis]:
        """Find analyses that took longer than threshold."""
        pass
    
    @abstractmethod
    async def get_provider_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics by provider."""
        pass


class AnalysisTaskRepository(Repository):
    """Repository interface for AnalysisTask aggregate."""
    
    @abstractmethod
    async def save(self, task: AnalysisTask) -> None:
        """Save analysis task to persistent storage."""
        pass
    
    @abstractmethod
    async def delete(self, task: AnalysisTask) -> None:
        """Delete analysis task from persistent storage."""
        pass
    
    @abstractmethod
    async def find_by_id(self, task_id: TaskId) -> Optional[AnalysisTask]:
        """Find analysis task by ID."""
        pass
    
    @abstractmethod
    async def find_by_upload_id(self, upload_id: UploadId) -> List[AnalysisTask]:
        """Find analysis tasks by upload ID."""
        pass
    
    @abstractmethod
    async def find_by_user_id(self, user_id: UserId) -> List[AnalysisTask]:
        """Find analysis tasks by user ID."""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: ProcessingStatus) -> List[AnalysisTask]:
        """Find analysis tasks by status."""
        pass
    
    @abstractmethod
    async def find_pending_tasks(self) -> List[AnalysisTask]:
        """Find all pending analysis tasks."""
        pass
    
    @abstractmethod
    async def find_in_progress_tasks(self) -> List[AnalysisTask]:
        """Find all in-progress analysis tasks."""
        pass
    
    @abstractmethod
    async def find_failed_tasks(self) -> List[AnalysisTask]:
        """Find all failed analysis tasks."""
        pass
    
    @abstractmethod
    async def find_tasks_for_retry(self) -> List[AnalysisTask]:
        """Find tasks that can be retried."""
        pass
    
    @abstractmethod
    async def find_high_priority_tasks(self) -> List[AnalysisTask]:
        """Find high priority analysis tasks."""
        pass
    
    @abstractmethod
    async def find_stale_tasks(self, timeout_minutes: int = 60) -> List[AnalysisTask]:
        """Find tasks that have been running too long."""
        pass
    
    @abstractmethod
    async def find_tasks_by_type(self, task_type: str) -> List[AnalysisTask]:
        """Find analysis tasks by type."""
        pass
    
    @abstractmethod
    async def count_tasks_by_status(self) -> Dict[ProcessingStatus, int]:
        """Count analysis tasks by status."""
        pass
    
    @abstractmethod
    async def count_tasks_by_user(self, user_id: UserId) -> int:
        """Count analysis tasks by user."""
        pass
    
    @abstractmethod
    async def find_next_task_to_process(self) -> Optional[AnalysisTask]:
        """Find the next task to process based on priority and creation time."""
        pass
    
    @abstractmethod
    async def cleanup_old_tasks(self, days_old: int = 30) -> int:
        """Clean up tasks older than specified days."""
        pass
    
    @abstractmethod
    async def get_task_statistics(self) -> Dict[str, Any]:
        """Get comprehensive task statistics."""
        pass


class EventRepository(ABC):
    """Repository interface for domain events."""
    
    @abstractmethod
    async def save_event(self, event: 'DomainEvent') -> None:
        """Save domain event to event store."""
        pass
    
    @abstractmethod
    async def find_events_by_aggregate_id(
        self, 
        aggregate_id: str, 
        from_version: int = 0
    ) -> List['DomainEvent']:
        """Find events by aggregate ID."""
        pass
    
    @abstractmethod
    async def find_events_by_type(self, event_type: str) -> List['DomainEvent']:
        """Find events by type."""
        pass
    
    @abstractmethod
    async def find_events_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List['DomainEvent']:
        """Find events within date range."""
        pass
    
    @abstractmethod
    async def find_unpublished_events(self) -> List['DomainEvent']:
        """Find events that haven't been published yet."""
        pass
    
    @abstractmethod
    async def mark_event_as_published(self, event_id: str) -> None:
        """Mark event as published."""
        pass
    
    @abstractmethod
    async def get_event_statistics(self) -> Dict[str, Any]:
        """Get event statistics."""
        pass


class UnitOfWork(ABC):
    """Unit of Work interface for managing transactions."""
    
    @abstractmethod
    async def __aenter__(self):
        """Enter async context manager."""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        pass
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit the transaction."""
        pass
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the transaction."""
        pass
    
    @property
    @abstractmethod
    def users(self) -> UserRepository:
        """Get user repository."""
        pass
    
    @property
    @abstractmethod
    def file_uploads(self) -> FileUploadRepository:
        """Get file upload repository."""
        pass
    
    @property
    @abstractmethod
    def sentiment_analyses(self) -> SentimentAnalysisRepository:
        """Get sentiment analysis repository."""
        pass
    
    @property
    @abstractmethod
    def analysis_tasks(self) -> AnalysisTaskRepository:
        """Get analysis task repository."""
        pass
    
    @property
    @abstractmethod
    def events(self) -> EventRepository:
        """Get event repository."""
        pass


class QueryRepository(ABC):
    """Base interface for read-only queries."""
    
    @abstractmethod
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute raw query."""
        pass


class AnalyticsQueryRepository(QueryRepository):
    """Repository for analytics queries."""
    
    @abstractmethod
    async def get_upload_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get upload trends over time."""
        pass
    
    @abstractmethod
    async def get_sentiment_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get sentiment trends over time."""
        pass
    
    @abstractmethod
    async def get_user_activity_summary(self, user_id: UserId) -> Dict[str, Any]:
        """Get user activity summary."""
        pass
    
    @abstractmethod
    async def get_processing_performance_metrics(self) -> Dict[str, Any]:
        """Get processing performance metrics."""
        pass
    
    @abstractmethod
    async def get_top_performing_providers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing sentiment analysis providers."""
        pass
    
    @abstractmethod
    async def get_error_analysis(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get error analysis for troubleshooting."""
        pass
    
    @abstractmethod
    async def get_capacity_planning_data(self) -> Dict[str, Any]:
        """Get data for capacity planning."""
        pass