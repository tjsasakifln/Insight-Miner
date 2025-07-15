"""
Domain entities following DDD principles with rich business logic.
"""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
from decimal import Decimal

from .value_objects import (
    UserId, Email, SentimentScore, ReviewText, FileName, 
    ProcessingStatus, AnalysisType, UploadId, TaskId
)
from .events import DomainEvent


class AggregateRoot(ABC):
    """Base class for aggregate roots with domain events."""
    
    def __init__(self):
        self._domain_events: List[DomainEvent] = []
        self._version = 0
    
    def raise_event(self, event: DomainEvent):
        """Raise a domain event."""
        self._domain_events.append(event)
    
    def clear_events(self):
        """Clear all domain events."""
        self._domain_events.clear()
    
    @property
    def domain_events(self) -> List[DomainEvent]:
        """Get all domain events."""
        return self._domain_events.copy()
    
    @property
    def version(self) -> int:
        """Get aggregate version for optimistic locking."""
        return self._version
    
    def increment_version(self):
        """Increment version for optimistic locking."""
        self._version += 1


@dataclass
class User(AggregateRoot):
    """User aggregate root with authentication and authorization."""
    
    id: UserId
    email: Email
    password_hash: str
    role: str
    is_active: bool = True
    is_verified: bool = False
    two_factor_enabled: bool = False
    two_factor_secret: Optional[str] = None
    failed_login_attempts: int = 0
    last_login: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        super().__init__()
        if not self.id:
            self.id = UserId(str(uuid.uuid4()))
    
    def authenticate(self, password: str, password_verifier) -> bool:
        """Authenticate user with password."""
        if not self.is_active:
            return False
        
        if password_verifier.verify(password, self.password_hash):
            self.failed_login_attempts = 0
            self.last_login = datetime.utcnow()
            self.updated_at = datetime.utcnow()
            return True
        
        self.failed_login_attempts += 1
        self.updated_at = datetime.utcnow()
        
        if self.failed_login_attempts >= 5:
            self.is_active = False
            from .events import UserAccountLockedEvent
            self.raise_event(UserAccountLockedEvent(
                user_id=self.id,
                email=self.email,
                failed_attempts=self.failed_login_attempts,
                occurred_at=datetime.utcnow()
            ))
        
        return False
    
    def enable_two_factor(self, secret: str):
        """Enable two-factor authentication."""
        self.two_factor_enabled = True
        self.two_factor_secret = secret
        self.updated_at = datetime.utcnow()
        
        from .events import TwoFactorEnabledEvent
        self.raise_event(TwoFactorEnabledEvent(
            user_id=self.id,
            email=self.email,
            occurred_at=datetime.utcnow()
        ))
    
    def disable_two_factor(self):
        """Disable two-factor authentication."""
        self.two_factor_enabled = False
        self.two_factor_secret = None
        self.updated_at = datetime.utcnow()
        
        from .events import TwoFactorDisabledEvent
        self.raise_event(TwoFactorDisabledEvent(
            user_id=self.id,
            email=self.email,
            occurred_at=datetime.utcnow()
        ))
    
    def verify_account(self):
        """Verify user account."""
        self.is_verified = True
        self.updated_at = datetime.utcnow()
        
        from .events import UserAccountVerifiedEvent
        self.raise_event(UserAccountVerifiedEvent(
            user_id=self.id,
            email=self.email,
            occurred_at=datetime.utcnow()
        ))
    
    def can_upload_file(self) -> bool:
        """Check if user can upload files."""
        return self.is_active and self.is_verified
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        role_permissions = {
            'admin': ['upload', 'analyze', 'manage_users', 'view_analytics'],
            'analyst': ['upload', 'analyze', 'view_analytics'],
            'viewer': ['view_analytics']
        }
        return permission in role_permissions.get(self.role, [])


@dataclass
class ReviewData:
    """Individual review data within an upload."""
    
    id: str
    text: ReviewText
    original_metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class FileUpload(AggregateRoot):
    """File upload aggregate with business rules and validation."""
    
    id: UploadId
    uploader_id: UserId
    file_name: FileName
    file_size: int
    file_path: str
    status: ProcessingStatus
    reviews: List[ReviewData] = field(default_factory=list)
    total_reviews: int = 0
    processed_reviews: int = 0
    failed_reviews: int = 0
    error_message: Optional[str] = None
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        super().__init__()
        if not self.id:
            self.id = UploadId(str(uuid.uuid4()))
    
    def start_processing(self):
        """Start processing the uploaded file."""
        if self.status != ProcessingStatus.PENDING:
            raise ValueError(f"Cannot start processing. Current status: {self.status}")
        
        self.status = ProcessingStatus.IN_PROGRESS
        self.processing_started_at = datetime.utcnow()
        self.increment_version()
        
        from .events import FileProcessingStartedEvent
        self.raise_event(FileProcessingStartedEvent(
            upload_id=self.id,
            uploader_id=self.uploader_id,
            file_name=self.file_name,
            total_reviews=self.total_reviews,
            occurred_at=datetime.utcnow()
        ))
    
    def complete_processing(self):
        """Complete processing successfully."""
        if self.status != ProcessingStatus.IN_PROGRESS:
            raise ValueError(f"Cannot complete processing. Current status: {self.status}")
        
        self.status = ProcessingStatus.COMPLETED
        self.processing_completed_at = datetime.utcnow()
        self.increment_version()
        
        from .events import FileProcessingCompletedEvent
        self.raise_event(FileProcessingCompletedEvent(
            upload_id=self.id,
            uploader_id=self.uploader_id,
            file_name=self.file_name,
            total_reviews=self.total_reviews,
            processed_reviews=self.processed_reviews,
            processing_time=self.processing_time,
            occurred_at=datetime.utcnow()
        ))
    
    def fail_processing(self, error_message: str):
        """Fail processing with error message."""
        if self.status != ProcessingStatus.IN_PROGRESS:
            raise ValueError(f"Cannot fail processing. Current status: {self.status}")
        
        self.status = ProcessingStatus.FAILED
        self.error_message = error_message
        self.processing_completed_at = datetime.utcnow()
        self.increment_version()
        
        from .events import FileProcessingFailedEvent
        self.raise_event(FileProcessingFailedEvent(
            upload_id=self.id,
            uploader_id=self.uploader_id,
            file_name=self.file_name,
            error_message=error_message,
            occurred_at=datetime.utcnow()
        ))
    
    def update_progress(self, processed_count: int, failed_count: int = 0):
        """Update processing progress."""
        if self.status != ProcessingStatus.IN_PROGRESS:
            raise ValueError(f"Cannot update progress. Current status: {self.status}")
        
        self.processed_reviews = processed_count
        self.failed_reviews = failed_count
        
        from .events import ProcessingProgressUpdatedEvent
        self.raise_event(ProcessingProgressUpdatedEvent(
            upload_id=self.id,
            processed_reviews=self.processed_reviews,
            total_reviews=self.total_reviews,
            failed_reviews=self.failed_reviews,
            progress_percentage=self.progress_percentage,
            occurred_at=datetime.utcnow()
        ))
    
    @property
    def progress_percentage(self) -> float:
        """Calculate processing progress percentage."""
        if self.total_reviews == 0:
            return 0.0
        return (self.processed_reviews / self.total_reviews) * 100
    
    @property
    def processing_time(self) -> Optional[timedelta]:
        """Calculate processing time."""
        if not self.processing_started_at:
            return None
        
        end_time = self.processing_completed_at or datetime.utcnow()
        return end_time - self.processing_started_at
    
    @property
    def is_completed(self) -> bool:
        """Check if processing is completed."""
        return self.status == ProcessingStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if processing failed."""
        return self.status == ProcessingStatus.FAILED
    
    @property
    def is_in_progress(self) -> bool:
        """Check if processing is in progress."""
        return self.status == ProcessingStatus.IN_PROGRESS
    
    def add_review(self, review_data: ReviewData):
        """Add review data to the upload."""
        self.reviews.append(review_data)
        self.total_reviews += 1
    
    def validate_file_size(self, max_size: int) -> bool:
        """Validate file size against maximum allowed."""
        return self.file_size <= max_size
    
    def validate_file_type(self, allowed_types: List[str]) -> bool:
        """Validate file type against allowed types."""
        file_extension = self.file_name.value.lower().split('.')[-1]
        return f".{file_extension}" in allowed_types


@dataclass
class SentimentAnalysis(AggregateRoot):
    """Sentiment analysis result aggregate."""
    
    id: str
    upload_id: UploadId
    review_id: str
    review_text: ReviewText
    sentiment_score: SentimentScore
    confidence: float
    provider: str
    analysis_type: AnalysisType
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        super().__init__()
        if not self.id:
            self.id = str(uuid.uuid4())
    
    @property
    def is_positive(self) -> bool:
        """Check if sentiment is positive."""
        return self.sentiment_score.value > 0.1
    
    @property
    def is_negative(self) -> bool:
        """Check if sentiment is negative."""
        return self.sentiment_score.value < -0.1
    
    @property
    def is_neutral(self) -> bool:
        """Check if sentiment is neutral."""
        return -0.1 <= self.sentiment_score.value <= 0.1
    
    @property
    def sentiment_label(self) -> str:
        """Get sentiment label."""
        if self.is_positive:
            return "positive"
        elif self.is_negative:
            return "negative"
        else:
            return "neutral"
    
    def validate_confidence(self) -> bool:
        """Validate confidence score."""
        return 0.0 <= self.confidence <= 1.0
    
    def validate_processing_time(self) -> bool:
        """Validate processing time."""
        return self.processing_time >= 0.0


@dataclass
class AnalysisTask(AggregateRoot):
    """Analysis task for tracking processing jobs."""
    
    id: TaskId
    upload_id: UploadId
    user_id: UserId
    task_type: str
    status: ProcessingStatus
    priority: int = 1
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        super().__init__()
        if not self.id:
            self.id = TaskId(str(uuid.uuid4()))
    
    def start_execution(self):
        """Start task execution."""
        if self.status != ProcessingStatus.PENDING:
            raise ValueError(f"Cannot start task. Current status: {self.status}")
        
        self.status = ProcessingStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
        self.increment_version()
        
        from .events import TaskStartedEvent
        self.raise_event(TaskStartedEvent(
            task_id=self.id,
            upload_id=self.upload_id,
            user_id=self.user_id,
            task_type=self.task_type,
            occurred_at=datetime.utcnow()
        ))
    
    def complete_execution(self):
        """Complete task execution."""
        if self.status != ProcessingStatus.IN_PROGRESS:
            raise ValueError(f"Cannot complete task. Current status: {self.status}")
        
        self.status = ProcessingStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.increment_version()
        
        from .events import TaskCompletedEvent
        self.raise_event(TaskCompletedEvent(
            task_id=self.id,
            upload_id=self.upload_id,
            user_id=self.user_id,
            task_type=self.task_type,
            execution_time=self.execution_time,
            occurred_at=datetime.utcnow()
        ))
    
    def fail_execution(self, error_message: str):
        """Fail task execution."""
        if self.status != ProcessingStatus.IN_PROGRESS:
            raise ValueError(f"Cannot fail task. Current status: {self.status}")
        
        self.error_message = error_message
        self.retry_count += 1
        
        if self.retry_count >= self.max_retries:
            self.status = ProcessingStatus.FAILED
            self.completed_at = datetime.utcnow()
            
            from .events import TaskFailedEvent
            self.raise_event(TaskFailedEvent(
                task_id=self.id,
                upload_id=self.upload_id,
                user_id=self.user_id,
                task_type=self.task_type,
                error_message=error_message,
                retry_count=self.retry_count,
                occurred_at=datetime.utcnow()
            ))
        else:
            self.status = ProcessingStatus.PENDING
            
            from .events import TaskRetryScheduledEvent
            self.raise_event(TaskRetryScheduledEvent(
                task_id=self.id,
                upload_id=self.upload_id,
                user_id=self.user_id,
                task_type=self.task_type,
                retry_count=self.retry_count,
                max_retries=self.max_retries,
                occurred_at=datetime.utcnow()
            ))
        
        self.increment_version()
    
    @property
    def execution_time(self) -> Optional[timedelta]:
        """Calculate task execution time."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        return end_time - self.started_at
    
    @property
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries
    
    @property
    def is_high_priority(self) -> bool:
        """Check if task has high priority."""
        return self.priority >= 5