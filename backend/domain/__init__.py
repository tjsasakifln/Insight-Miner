"""
Domain layer containing business logic, entities, and domain services.
"""

from .entities import User, FileUpload, SentimentAnalysis, AnalysisTask, AggregateRoot
from .value_objects import (
    UserId, Email, SentimentScore, ReviewText, FileName, 
    ProcessingStatus, AnalysisType, UploadId, TaskId, FileSize, 
    ProcessingTime, Confidence, Priority, RetryCount
)
from .events import (
    DomainEvent, UserRegisteredEvent, UserAccountVerifiedEvent,
    FileUploadedEvent, FileProcessingStartedEvent, FileProcessingCompletedEvent,
    SentimentAnalysisCompletedEvent, TaskCreatedEvent, TaskCompletedEvent
)
from .services import (
    FileValidationService, SentimentAnalysisService, UserPermissionService,
    BusinessRuleService, MetricsCalculationService, ValidationResult, AnalysisMetrics
)
from .repositories import (
    Repository, UserRepository, FileUploadRepository, SentimentAnalysisRepository,
    AnalysisTaskRepository, EventRepository, UnitOfWork, QueryRepository,
    AnalyticsQueryRepository
)

__all__ = [
    # Entities
    "User", "FileUpload", "SentimentAnalysis", "AnalysisTask", "AggregateRoot",
    
    # Value Objects
    "UserId", "Email", "SentimentScore", "ReviewText", "FileName",
    "ProcessingStatus", "AnalysisType", "UploadId", "TaskId", "FileSize",
    "ProcessingTime", "Confidence", "Priority", "RetryCount",
    
    # Events
    "DomainEvent", "UserRegisteredEvent", "UserAccountVerifiedEvent",
    "FileUploadedEvent", "FileProcessingStartedEvent", "FileProcessingCompletedEvent",
    "SentimentAnalysisCompletedEvent", "TaskCreatedEvent", "TaskCompletedEvent",
    
    # Services
    "FileValidationService", "SentimentAnalysisService", "UserPermissionService",
    "BusinessRuleService", "MetricsCalculationService", "ValidationResult", "AnalysisMetrics",
    
    # Repositories
    "Repository", "UserRepository", "FileUploadRepository", "SentimentAnalysisRepository",
    "AnalysisTaskRepository", "EventRepository", "UnitOfWork", "QueryRepository",
    "AnalyticsQueryRepository"
]