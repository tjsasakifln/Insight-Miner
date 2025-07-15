"""
Application layer use cases implementing business workflows.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

from ..domain.entities import User, FileUpload, SentimentAnalysis, AnalysisTask, ReviewData
from ..domain.value_objects import (
    UserId, Email, UploadId, TaskId, FileName, SentimentScore, 
    ReviewText, ProcessingStatus, AnalysisType, Priority
)
from ..domain.events import (
    UserRegisteredEvent, FileUploadedEvent, FileProcessingStartedEvent,
    TaskCreatedEvent, DomainEvent
)
from ..domain.repositories import UnitOfWork, AnalyticsQueryRepository
from ..domain.services import (
    FileValidationService, SentimentAnalysisService, 
    UserPermissionService, BusinessRuleService, MetricsCalculationService
)
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class UseCaseResult:
    """Result of a use case execution."""
    success: bool
    data: Any = None
    error_message: str = ""
    validation_errors: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []


class UseCase(ABC):
    """Base class for all use cases."""
    
    def __init__(self, uow: UnitOfWork):
        self.uow = uow
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    async def execute(self, request: Any) -> UseCaseResult:
        """Execute the use case."""
        pass
    
    async def publish_events(self, events: List[DomainEvent]):
        """Publish domain events."""
        for event in events:
            await self.uow.events.save_event(event)
            self.logger.info(f"Published event: {event.event_type}", event_id=event.event_id)


# User Management Use Cases

@dataclass
class RegisterUserRequest:
    """Request for registering a new user."""
    email: str
    password: str
    role: str = "analyst"


class RegisterUserUseCase(UseCase):
    """Use case for registering a new user."""
    
    def __init__(self, uow: UnitOfWork):
        super().__init__(uow)
        self.permission_service = UserPermissionService()
    
    async def execute(self, request: RegisterUserRequest) -> UseCaseResult:
        """Execute user registration."""
        try:
            # Validate email format
            try:
                email = Email(request.email)
            except ValueError as e:
                return UseCaseResult(success=False, error_message=str(e))
            
            # Check if user already exists
            async with self.uow:
                existing_user = await self.uow.users.find_by_email(email)
                if existing_user:
                    return UseCaseResult(success=False, error_message="User with this email already exists")
                
                # Create new user
                user = User(
                    id=UserId(str(uuid.uuid4())),
                    email=email,
                    password_hash=request.password,  # Should be hashed by auth service
                    role=request.role,
                    created_at=datetime.utcnow()
                )
                
                await self.uow.users.save(user)
                
                # Publish domain event
                await self.publish_events(user.domain_events)
                user.clear_events()
                
                await self.uow.commit()
                
                self.logger.info(f"User registered successfully", user_id=str(user.id), email=str(email))
                
                return UseCaseResult(
                    success=True,
                    data={
                        "user_id": str(user.id),
                        "email": str(user.email),
                        "role": user.role
                    }
                )
        
        except Exception as e:
            await self.uow.rollback()
            self.logger.error(f"User registration failed", error=str(e))
            return UseCaseResult(success=False, error_message=f"Registration failed: {str(e)}")


@dataclass
class AuthenticateUserRequest:
    """Request for user authentication."""
    email: str
    password: str
    two_factor_token: Optional[str] = None


class AuthenticateUserUseCase(UseCase):
    """Use case for user authentication."""
    
    async def execute(self, request: AuthenticateUserRequest) -> UseCaseResult:
        """Execute user authentication."""
        try:
            email = Email(request.email)
            
            async with self.uow:
                user = await self.uow.users.find_by_email(email)
                if not user:
                    return UseCaseResult(success=False, error_message="Invalid credentials")
                
                # Authenticate user (password verification would be done by auth service)
                if user.authenticate(request.password, None):  # Pass proper password verifier
                    # Handle 2FA if enabled
                    if user.two_factor_enabled:
                        if not request.two_factor_token:
                            return UseCaseResult(success=False, error_message="Two-factor token required")
                        
                        # Verify 2FA token (would be done by auth service)
                        # if not verify_2fa_token(user.two_factor_secret, request.two_factor_token):
                        #     return UseCaseResult(success=False, error_message="Invalid two-factor token")
                    
                    await self.uow.users.save(user)
                    await self.uow.commit()
                    
                    self.logger.info(f"User authenticated successfully", user_id=str(user.id))
                    
                    return UseCaseResult(
                        success=True,
                        data={
                            "user_id": str(user.id),
                            "email": str(user.email),
                            "role": user.role,
                            "last_login": user.last_login.isoformat() if user.last_login else None
                        }
                    )
                else:
                    await self.uow.users.save(user)
                    await self.uow.commit()
                    
                    return UseCaseResult(success=False, error_message="Invalid credentials")
        
        except Exception as e:
            await self.uow.rollback()
            self.logger.error(f"Authentication failed", error=str(e))
            return UseCaseResult(success=False, error_message="Authentication failed")


# File Upload Use Cases

@dataclass
class UploadFileRequest:
    """Request for uploading a file."""
    user_id: str
    file_name: str
    file_content: bytes
    file_size: int


class UploadFileUseCase(UseCase):
    """Use case for uploading a file."""
    
    def __init__(self, uow: UnitOfWork):
        super().__init__(uow)
        self.file_validation_service = FileValidationService()
        self.permission_service = UserPermissionService()
        self.business_rule_service = BusinessRuleService()
    
    async def execute(self, request: UploadFileRequest) -> UseCaseResult:
        """Execute file upload."""
        try:
            user_id = UserId(request.user_id)
            
            async with self.uow:
                # Check if user exists and has permission
                user = await self.uow.users.find_by_id(user_id)
                if not user:
                    return UseCaseResult(success=False, error_message="User not found")
                
                if not user.can_upload_file():
                    return UseCaseResult(success=False, error_message="User cannot upload files")
                
                # Check business rules
                active_uploads = await self.uow.file_uploads.find_by_status(ProcessingStatus.IN_PROGRESS)
                can_upload, reason = self.business_rule_service.can_user_upload_file(user, active_uploads)
                if not can_upload:
                    return UseCaseResult(success=False, error_message=reason)
                
                # Create file upload entity
                file_upload = FileUpload(
                    id=UploadId(str(uuid.uuid4())),
                    uploader_id=user_id,
                    file_name=FileName(request.file_name),
                    file_size=request.file_size,
                    file_path=f"/uploads/{uuid.uuid4()}_{request.file_name}",
                    status=ProcessingStatus.PENDING,
                    uploaded_at=datetime.utcnow()
                )
                
                # Validate file upload
                validation_result = self.file_validation_service.validate_file_upload(file_upload)
                if not validation_result.is_valid:
                    return UseCaseResult(
                        success=False,
                        error_message="File validation failed",
                        validation_errors=validation_result.errors
                    )
                
                # Save file upload
                await self.uow.file_uploads.save(file_upload)
                
                # Publish domain events
                await self.publish_events(file_upload.domain_events)
                file_upload.clear_events()
                
                await self.uow.commit()
                
                self.logger.info(f"File uploaded successfully", 
                               upload_id=str(file_upload.id), 
                               user_id=str(user_id))
                
                return UseCaseResult(
                    success=True,
                    data={
                        "upload_id": str(file_upload.id),
                        "file_name": str(file_upload.file_name),
                        "file_size": file_upload.file_size,
                        "status": str(file_upload.status)
                    }
                )
        
        except Exception as e:
            await self.uow.rollback()
            self.logger.error(f"File upload failed", error=str(e))
            return UseCaseResult(success=False, error_message=f"Upload failed: {str(e)}")


@dataclass
class ProcessFileRequest:
    """Request for processing an uploaded file."""
    upload_id: str
    user_id: str


class ProcessFileUseCase(UseCase):
    """Use case for processing an uploaded file."""
    
    def __init__(self, uow: UnitOfWork):
        super().__init__(uow)
        self.business_rule_service = BusinessRuleService()
    
    async def execute(self, request: ProcessFileRequest) -> UseCaseResult:
        """Execute file processing."""
        try:
            upload_id = UploadId(request.upload_id)
            user_id = UserId(request.user_id)
            
            async with self.uow:
                # Get file upload
                file_upload = await self.uow.file_uploads.find_by_id(upload_id)
                if not file_upload:
                    return UseCaseResult(success=False, error_message="File upload not found")
                
                # Check user permission
                user = await self.uow.users.find_by_id(user_id)
                if not user:
                    return UseCaseResult(success=False, error_message="User not found")
                
                if not self.permission_service.can_user_access_upload(user, file_upload):
                    return UseCaseResult(success=False, error_message="Access denied")
                
                # Validate file for processing
                validation_result = self.business_rule_service.validate_file_for_processing(file_upload)
                if not validation_result.is_valid:
                    return UseCaseResult(
                        success=False,
                        error_message="File validation failed",
                        validation_errors=validation_result.errors
                    )
                
                # Start processing
                file_upload.start_processing()
                
                # Calculate priority
                priority = self.business_rule_service.calculate_processing_priority(user, file_upload)
                
                # Create analysis task
                analysis_task = AnalysisTask(
                    id=TaskId(str(uuid.uuid4())),
                    upload_id=upload_id,
                    user_id=user_id,
                    task_type="sentiment_analysis",
                    status=ProcessingStatus.PENDING,
                    priority=priority.value,
                    created_at=datetime.utcnow()
                )
                
                # Save entities
                await self.uow.file_uploads.save(file_upload)
                await self.uow.analysis_tasks.save(analysis_task)
                
                # Publish domain events
                await self.publish_events(file_upload.domain_events)
                await self.publish_events(analysis_task.domain_events)
                
                file_upload.clear_events()
                analysis_task.clear_events()
                
                await self.uow.commit()
                
                self.logger.info(f"File processing started", 
                               upload_id=str(upload_id), 
                               task_id=str(analysis_task.id))
                
                return UseCaseResult(
                    success=True,
                    data={
                        "upload_id": str(upload_id),
                        "task_id": str(analysis_task.id),
                        "status": str(file_upload.status),
                        "priority": priority.value
                    }
                )
        
        except Exception as e:
            await self.uow.rollback()
            self.logger.error(f"File processing failed", error=str(e))
            return UseCaseResult(success=False, error_message=f"Processing failed: {str(e)}")


# Analysis Use Cases

@dataclass
class GetSentimentAnalysisRequest:
    """Request for getting sentiment analysis results."""
    upload_id: str
    user_id: str


class GetSentimentAnalysisUseCase(UseCase):
    """Use case for getting sentiment analysis results."""
    
    def __init__(self, uow: UnitOfWork):
        super().__init__(uow)
        self.permission_service = UserPermissionService()
        self.sentiment_service = SentimentAnalysisService()
    
    async def execute(self, request: GetSentimentAnalysisRequest) -> UseCaseResult:
        """Execute sentiment analysis retrieval."""
        try:
            upload_id = UploadId(request.upload_id)
            user_id = UserId(request.user_id)
            
            async with self.uow:
                # Check user permission
                user = await self.uow.users.find_by_id(user_id)
                if not user:
                    return UseCaseResult(success=False, error_message="User not found")
                
                file_upload = await self.uow.file_uploads.find_by_id(upload_id)
                if not file_upload:
                    return UseCaseResult(success=False, error_message="File upload not found")
                
                if not self.permission_service.can_user_access_upload(user, file_upload):
                    return UseCaseResult(success=False, error_message="Access denied")
                
                # Get sentiment analyses
                analyses = await self.uow.sentiment_analyses.find_by_upload_id(upload_id)
                
                # Calculate metrics
                sentiment_distribution = self.sentiment_service.calculate_sentiment_distribution(analyses)
                average_sentiment = self.sentiment_service.calculate_average_sentiment(analyses)
                trends = self.sentiment_service.identify_sentiment_trends(analyses)
                anomalies = self.sentiment_service.detect_sentiment_anomalies(analyses)
                
                self.logger.info(f"Sentiment analysis retrieved", 
                               upload_id=str(upload_id), 
                               analysis_count=len(analyses))
                
                return UseCaseResult(
                    success=True,
                    data={
                        "upload_id": str(upload_id),
                        "total_analyses": len(analyses),
                        "sentiment_distribution": sentiment_distribution,
                        "average_sentiment": average_sentiment.value if average_sentiment else None,
                        "trends": trends,
                        "anomalies": anomalies,
                        "analyses": [
                            {
                                "id": analysis.id,
                                "review_id": analysis.review_id,
                                "sentiment_score": analysis.sentiment_score.value,
                                "confidence": analysis.confidence,
                                "provider": analysis.provider,
                                "sentiment_label": analysis.sentiment_label,
                                "analyzed_at": analysis.analyzed_at.isoformat()
                            }
                            for analysis in analyses
                        ]
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Failed to get sentiment analysis", error=str(e))
            return UseCaseResult(success=False, error_message=f"Failed to get analysis: {str(e)}")


@dataclass
class GetUserUploadsRequest:
    """Request for getting user uploads."""
    user_id: str
    limit: int = 50


class GetUserUploadsUseCase(UseCase):
    """Use case for getting user uploads."""
    
    async def execute(self, request: GetUserUploadsRequest) -> UseCaseResult:
        """Execute user uploads retrieval."""
        try:
            user_id = UserId(request.user_id)
            
            async with self.uow:
                # Check if user exists
                user = await self.uow.users.find_by_id(user_id)
                if not user:
                    return UseCaseResult(success=False, error_message="User not found")
                
                # Get user uploads
                uploads = await self.uow.file_uploads.find_by_uploader_id(user_id)
                
                # Sort by upload date (most recent first)
                uploads.sort(key=lambda x: x.uploaded_at, reverse=True)
                
                # Apply limit
                uploads = uploads[:request.limit]
                
                self.logger.info(f"User uploads retrieved", 
                               user_id=str(user_id), 
                               upload_count=len(uploads))
                
                return UseCaseResult(
                    success=True,
                    data={
                        "user_id": str(user_id),
                        "uploads": [
                            {
                                "id": str(upload.id),
                                "file_name": str(upload.file_name),
                                "file_size": upload.file_size,
                                "status": str(upload.status),
                                "total_reviews": upload.total_reviews,
                                "processed_reviews": upload.processed_reviews,
                                "progress_percentage": upload.progress_percentage,
                                "uploaded_at": upload.uploaded_at.isoformat(),
                                "processing_time": upload.processing_time.total_seconds() if upload.processing_time else None
                            }
                            for upload in uploads
                        ]
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Failed to get user uploads", error=str(e))
            return UseCaseResult(success=False, error_message=f"Failed to get uploads: {str(e)}")


# Analytics Use Cases

@dataclass
class GetAnalyticsRequest:
    """Request for getting analytics."""
    user_id: str
    days: int = 30


class GetAnalyticsUseCase(UseCase):
    """Use case for getting analytics."""
    
    def __init__(self, uow: UnitOfWork, analytics_repo: AnalyticsQueryRepository):
        super().__init__(uow)
        self.analytics_repo = analytics_repo
        self.metrics_service = MetricsCalculationService()
    
    async def execute(self, request: GetAnalyticsRequest) -> UseCaseResult:
        """Execute analytics retrieval."""
        try:
            user_id = UserId(request.user_id)
            
            async with self.uow:
                # Check user permission
                user = await self.uow.users.find_by_id(user_id)
                if not user:
                    return UseCaseResult(success=False, error_message="User not found")
                
                if not user.has_permission("view_analytics"):
                    return UseCaseResult(success=False, error_message="Access denied")
                
                # Get analytics data
                upload_trends = await self.analytics_repo.get_upload_trends(request.days)
                sentiment_trends = await self.analytics_repo.get_sentiment_trends(request.days)
                processing_metrics = await self.analytics_repo.get_processing_performance_metrics()
                top_providers = await self.analytics_repo.get_top_performing_providers()
                
                # Get user-specific data
                user_activity = await self.analytics_repo.get_user_activity_summary(user_id)
                
                self.logger.info(f"Analytics retrieved", 
                               user_id=str(user_id), 
                               days=request.days)
                
                return UseCaseResult(
                    success=True,
                    data={
                        "user_id": str(user_id),
                        "period_days": request.days,
                        "upload_trends": upload_trends,
                        "sentiment_trends": sentiment_trends,
                        "processing_metrics": processing_metrics,
                        "top_providers": top_providers,
                        "user_activity": user_activity
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Failed to get analytics", error=str(e))
            return UseCaseResult(success=False, error_message=f"Failed to get analytics: {str(e)}")


# System Administration Use Cases

@dataclass
class GetSystemHealthRequest:
    """Request for getting system health."""
    user_id: str


class GetSystemHealthUseCase(UseCase):
    """Use case for getting system health."""
    
    async def execute(self, request: GetSystemHealthRequest) -> UseCaseResult:
        """Execute system health check."""
        try:
            user_id = UserId(request.user_id)
            
            async with self.uow:
                # Check user permission
                user = await self.uow.users.find_by_id(user_id)
                if not user:
                    return UseCaseResult(success=False, error_message="User not found")
                
                if not user.has_permission("system_admin"):
                    return UseCaseResult(success=False, error_message="Access denied")
                
                # Get system health data
                upload_stats = await self.uow.file_uploads.get_upload_statistics()
                task_stats = await self.uow.analysis_tasks.get_task_statistics()
                
                # Calculate health metrics
                health_score = self._calculate_health_score(upload_stats, task_stats)
                
                self.logger.info(f"System health retrieved", 
                               user_id=str(user_id), 
                               health_score=health_score)
                
                return UseCaseResult(
                    success=True,
                    data={
                        "health_score": health_score,
                        "upload_statistics": upload_stats,
                        "task_statistics": task_stats,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Failed to get system health", error=str(e))
            return UseCaseResult(success=False, error_message=f"Failed to get system health: {str(e)}")
    
    def _calculate_health_score(self, upload_stats: Dict, task_stats: Dict) -> float:
        """Calculate overall system health score."""
        # Simple health score calculation
        # In a real system, this would be more sophisticated
        score = 100.0
        
        # Penalize for high failure rates
        if upload_stats.get('failure_rate', 0) > 10:
            score -= 20
        
        if task_stats.get('failure_rate', 0) > 5:
            score -= 15
        
        # Penalize for long processing times
        if upload_stats.get('average_processing_time', 0) > 300:  # 5 minutes
            score -= 10
        
        return max(0, score)