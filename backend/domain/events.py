"""
Domain events for event-driven architecture and eventual consistency.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
import uuid

from .value_objects import (
    UserId, Email, UploadId, TaskId, FileName, 
    ProcessingTime, SentimentScore, AnalysisType
)


@dataclass
class DomainEvent(ABC):
    """Base class for all domain events."""
    
    event_id: str
    occurred_at: datetime
    event_version: int = 1
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        
        if not self.occurred_at:
            self.occurred_at = datetime.utcnow()
    
    @property
    @abstractmethod
    def event_type(self) -> str:
        """Get the event type identifier."""
        pass
    
    @property
    def aggregate_id(self) -> str:
        """Get the aggregate ID this event belongs to."""
        return getattr(self, 'user_id', getattr(self, 'upload_id', self.event_id))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'occurred_at': self.occurred_at.isoformat(),
            'event_version': self.event_version,
            'data': self._get_event_data()
        }
    
    @abstractmethod
    def _get_event_data(self) -> Dict[str, Any]:
        """Get event-specific data."""
        pass


# User Domain Events

@dataclass
class UserRegisteredEvent(DomainEvent):
    """Event raised when a user registers."""
    
    user_id: UserId
    email: Email
    role: str
    
    @property
    def event_type(self) -> str:
        return "user.registered"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'user_id': str(self.user_id),
            'email': str(self.email),
            'role': self.role
        }


@dataclass
class UserAccountVerifiedEvent(DomainEvent):
    """Event raised when a user account is verified."""
    
    user_id: UserId
    email: Email
    
    @property
    def event_type(self) -> str:
        return "user.account_verified"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'user_id': str(self.user_id),
            'email': str(self.email)
        }


@dataclass
class UserAccountLockedEvent(DomainEvent):
    """Event raised when a user account is locked."""
    
    user_id: UserId
    email: Email
    failed_attempts: int
    
    @property
    def event_type(self) -> str:
        return "user.account_locked"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'user_id': str(self.user_id),
            'email': str(self.email),
            'failed_attempts': self.failed_attempts
        }


@dataclass
class TwoFactorEnabledEvent(DomainEvent):
    """Event raised when two-factor authentication is enabled."""
    
    user_id: UserId
    email: Email
    
    @property
    def event_type(self) -> str:
        return "user.two_factor_enabled"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'user_id': str(self.user_id),
            'email': str(self.email)
        }


@dataclass
class TwoFactorDisabledEvent(DomainEvent):
    """Event raised when two-factor authentication is disabled."""
    
    user_id: UserId
    email: Email
    
    @property
    def event_type(self) -> str:
        return "user.two_factor_disabled"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'user_id': str(self.user_id),
            'email': str(self.email)
        }


@dataclass
class UserLoginEvent(DomainEvent):
    """Event raised when a user logs in."""
    
    user_id: UserId
    email: Email
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    @property
    def event_type(self) -> str:
        return "user.login"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'user_id': str(self.user_id),
            'email': str(self.email),
            'ip_address': self.ip_address,
            'user_agent': self.user_agent
        }


# File Upload Domain Events

@dataclass
class FileUploadedEvent(DomainEvent):
    """Event raised when a file is uploaded."""
    
    upload_id: UploadId
    uploader_id: UserId
    file_name: FileName
    file_size: int
    
    @property
    def event_type(self) -> str:
        return "file.uploaded"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'upload_id': str(self.upload_id),
            'uploader_id': str(self.uploader_id),
            'file_name': str(self.file_name),
            'file_size': self.file_size
        }


@dataclass
class FileProcessingStartedEvent(DomainEvent):
    """Event raised when file processing starts."""
    
    upload_id: UploadId
    uploader_id: UserId
    file_name: FileName
    total_reviews: int
    
    @property
    def event_type(self) -> str:
        return "file.processing_started"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'upload_id': str(self.upload_id),
            'uploader_id': str(self.uploader_id),
            'file_name': str(self.file_name),
            'total_reviews': self.total_reviews
        }


@dataclass
class FileProcessingCompletedEvent(DomainEvent):
    """Event raised when file processing is completed."""
    
    upload_id: UploadId
    uploader_id: UserId
    file_name: FileName
    total_reviews: int
    processed_reviews: int
    processing_time: ProcessingTime
    
    @property
    def event_type(self) -> str:
        return "file.processing_completed"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'upload_id': str(self.upload_id),
            'uploader_id': str(self.uploader_id),
            'file_name': str(self.file_name),
            'total_reviews': self.total_reviews,
            'processed_reviews': self.processed_reviews,
            'processing_time': self.processing_time.value
        }


@dataclass
class FileProcessingFailedEvent(DomainEvent):
    """Event raised when file processing fails."""
    
    upload_id: UploadId
    uploader_id: UserId
    file_name: FileName
    error_message: str
    
    @property
    def event_type(self) -> str:
        return "file.processing_failed"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'upload_id': str(self.upload_id),
            'uploader_id': str(self.uploader_id),
            'file_name': str(self.file_name),
            'error_message': self.error_message
        }


@dataclass
class ProcessingProgressUpdatedEvent(DomainEvent):
    """Event raised when processing progress is updated."""
    
    upload_id: UploadId
    processed_reviews: int
    total_reviews: int
    failed_reviews: int
    progress_percentage: float
    
    @property
    def event_type(self) -> str:
        return "file.progress_updated"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'upload_id': str(self.upload_id),
            'processed_reviews': self.processed_reviews,
            'total_reviews': self.total_reviews,
            'failed_reviews': self.failed_reviews,
            'progress_percentage': self.progress_percentage
        }


# Analysis Domain Events

@dataclass
class SentimentAnalysisCompletedEvent(DomainEvent):
    """Event raised when sentiment analysis is completed."""
    
    upload_id: UploadId
    review_id: str
    sentiment_score: SentimentScore
    confidence: float
    provider: str
    processing_time: ProcessingTime
    
    @property
    def event_type(self) -> str:
        return "analysis.sentiment_completed"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'upload_id': str(self.upload_id),
            'review_id': self.review_id,
            'sentiment_score': self.sentiment_score.value,
            'confidence': self.confidence,
            'provider': self.provider,
            'processing_time': self.processing_time.value
        }


@dataclass
class SentimentAnalysisFailedEvent(DomainEvent):
    """Event raised when sentiment analysis fails."""
    
    upload_id: UploadId
    review_id: str
    error_message: str
    provider: str
    
    @property
    def event_type(self) -> str:
        return "analysis.sentiment_failed"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'upload_id': str(self.upload_id),
            'review_id': self.review_id,
            'error_message': self.error_message,
            'provider': self.provider
        }


@dataclass
class BatchAnalysisCompletedEvent(DomainEvent):
    """Event raised when batch analysis is completed."""
    
    upload_id: UploadId
    batch_id: str
    analysis_type: AnalysisType
    batch_size: int
    successful_analyses: int
    failed_analyses: int
    processing_time: ProcessingTime
    
    @property
    def event_type(self) -> str:
        return "analysis.batch_completed"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'upload_id': str(self.upload_id),
            'batch_id': self.batch_id,
            'analysis_type': str(self.analysis_type),
            'batch_size': self.batch_size,
            'successful_analyses': self.successful_analyses,
            'failed_analyses': self.failed_analyses,
            'processing_time': self.processing_time.value
        }


# Task Domain Events

@dataclass
class TaskCreatedEvent(DomainEvent):
    """Event raised when a task is created."""
    
    task_id: TaskId
    upload_id: UploadId
    user_id: UserId
    task_type: str
    priority: int
    
    @property
    def event_type(self) -> str:
        return "task.created"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'task_id': str(self.task_id),
            'upload_id': str(self.upload_id),
            'user_id': str(self.user_id),
            'task_type': self.task_type,
            'priority': self.priority
        }


@dataclass
class TaskStartedEvent(DomainEvent):
    """Event raised when a task starts execution."""
    
    task_id: TaskId
    upload_id: UploadId
    user_id: UserId
    task_type: str
    
    @property
    def event_type(self) -> str:
        return "task.started"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'task_id': str(self.task_id),
            'upload_id': str(self.upload_id),
            'user_id': str(self.user_id),
            'task_type': self.task_type
        }


@dataclass
class TaskCompletedEvent(DomainEvent):
    """Event raised when a task is completed."""
    
    task_id: TaskId
    upload_id: UploadId
    user_id: UserId
    task_type: str
    execution_time: ProcessingTime
    
    @property
    def event_type(self) -> str:
        return "task.completed"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'task_id': str(self.task_id),
            'upload_id': str(self.upload_id),
            'user_id': str(self.user_id),
            'task_type': self.task_type,
            'execution_time': self.execution_time.value
        }


@dataclass
class TaskFailedEvent(DomainEvent):
    """Event raised when a task fails."""
    
    task_id: TaskId
    upload_id: UploadId
    user_id: UserId
    task_type: str
    error_message: str
    retry_count: int
    
    @property
    def event_type(self) -> str:
        return "task.failed"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'task_id': str(self.task_id),
            'upload_id': str(self.upload_id),
            'user_id': str(self.user_id),
            'task_type': self.task_type,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }


@dataclass
class TaskRetryScheduledEvent(DomainEvent):
    """Event raised when a task is scheduled for retry."""
    
    task_id: TaskId
    upload_id: UploadId
    user_id: UserId
    task_type: str
    retry_count: int
    max_retries: int
    
    @property
    def event_type(self) -> str:
        return "task.retry_scheduled"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'task_id': str(self.task_id),
            'upload_id': str(self.upload_id),
            'user_id': str(self.user_id),
            'task_type': self.task_type,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


# System Domain Events

@dataclass
class SystemHealthCheckEvent(DomainEvent):
    """Event raised for system health monitoring."""
    
    service_name: str
    status: str
    response_time: float
    metadata: Dict[str, Any]
    
    @property
    def event_type(self) -> str:
        return "system.health_check"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'service_name': self.service_name,
            'status': self.status,
            'response_time': self.response_time,
            'metadata': self.metadata
        }


@dataclass
class SystemErrorEvent(DomainEvent):
    """Event raised when a system error occurs."""
    
    service_name: str
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    @property
    def event_type(self) -> str:
        return "system.error"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'service_name': self.service_name,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'context': self.context or {}
        }


@dataclass
class SystemMetricsEvent(DomainEvent):
    """Event raised for system metrics reporting."""
    
    service_name: str
    metric_name: str
    metric_value: float
    metric_type: str
    tags: Dict[str, str]
    
    @property
    def event_type(self) -> str:
        return "system.metrics"
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'service_name': self.service_name,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_type': self.metric_type,
            'tags': self.tags
        }