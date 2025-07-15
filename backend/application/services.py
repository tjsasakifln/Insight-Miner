"""
Application services for orchestrating use cases and external integrations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from ..domain.entities import User, FileUpload, SentimentAnalysis, AnalysisTask
from ..domain.value_objects import UserId, UploadId, TaskId, ProcessingStatus
from ..domain.events import DomainEvent
from ..domain.repositories import UnitOfWork, AnalyticsQueryRepository
from ..logger import get_logger
from .use_cases import (
    RegisterUserUseCase, AuthenticateUserUseCase, UploadFileUseCase,
    ProcessFileUseCase, GetSentimentAnalysisUseCase, GetUserUploadsUseCase,
    GetAnalyticsUseCase, GetSystemHealthUseCase, UseCaseResult
)

logger = get_logger(__name__)


@dataclass
class ServiceResult:
    """Result of a service operation."""
    success: bool
    data: Any = None
    error_message: str = ""
    status_code: int = 200
    
    @classmethod
    def success_result(cls, data: Any = None, status_code: int = 200) -> 'ServiceResult':
        """Create a success result."""
        return cls(success=True, data=data, status_code=status_code)
    
    @classmethod
    def error_result(cls, error_message: str, status_code: int = 400) -> 'ServiceResult':
        """Create an error result."""
        return cls(success=False, error_message=error_message, status_code=status_code)


class EventPublisher(ABC):
    """Abstract event publisher for domain events."""
    
    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish a domain event."""
        pass


class EmailNotificationService(ABC):
    """Abstract email notification service."""
    
    @abstractmethod
    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email notification."""
        pass


class FileStorageService(ABC):
    """Abstract file storage service."""
    
    @abstractmethod
    async def save_file(self, file_path: str, content: bytes) -> bool:
        """Save file to storage."""
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from storage."""
        pass
    
    @abstractmethod
    async def get_file_url(self, file_path: str) -> str:
        """Get file URL."""
        pass


class CacheService(ABC):
    """Abstract cache service."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass


class TaskQueueService(ABC):
    """Abstract task queue service."""
    
    @abstractmethod
    async def enqueue_task(self, task_name: str, task_data: Dict[str, Any]) -> str:
        """Enqueue a task for processing."""
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        pass


class UserService:
    """Application service for user management."""
    
    def __init__(
        self,
        uow: UnitOfWork,
        event_publisher: EventPublisher,
        email_service: EmailNotificationService,
        cache_service: CacheService
    ):
        self.uow = uow
        self.event_publisher = event_publisher
        self.email_service = email_service
        self.cache_service = cache_service
        self.logger = get_logger(__name__)
    
    async def register_user(self, email: str, password: str, role: str = "analyst") -> ServiceResult:
        """Register a new user."""
        try:
            use_case = RegisterUserUseCase(self.uow)
            result = await use_case.execute(RegisterUserUseCase.RegisterUserRequest(
                email=email,
                password=password,
                role=role
            ))
            
            if result.success:
                # Send welcome email
                await self.email_service.send_email(
                    to=email,
                    subject="Welcome to Insight Miner",
                    body=f"Welcome to Insight Miner! Your account has been created successfully."
                )
                
                # Publish events
                await self._publish_domain_events()
                
                return ServiceResult.success_result(result.data, 201)
            else:
                return ServiceResult.error_result(
                    result.error_message,
                    400 if result.validation_errors else 422
                )
        
        except Exception as e:
            self.logger.error(f"User registration failed", error=str(e))
            return ServiceResult.error_result("Registration failed", 500)
    
    async def authenticate_user(self, email: str, password: str, two_factor_token: str = None) -> ServiceResult:
        """Authenticate a user."""
        try:
            use_case = AuthenticateUserUseCase(self.uow)
            result = await use_case.execute(AuthenticateUserUseCase.AuthenticateUserRequest(
                email=email,
                password=password,
                two_factor_token=two_factor_token
            ))
            
            if result.success:
                # Cache user session
                await self.cache_service.set(
                    f"user_session:{result.data['user_id']}",
                    result.data,
                    ttl=3600
                )
                
                return ServiceResult.success_result(result.data)
            else:
                return ServiceResult.error_result(result.error_message, 401)
        
        except Exception as e:
            self.logger.error(f"Authentication failed", error=str(e))
            return ServiceResult.error_result("Authentication failed", 500)
    
    async def get_user_profile(self, user_id: str) -> ServiceResult:
        """Get user profile."""
        try:
            # Try cache first
            cached_profile = await self.cache_service.get(f"user_profile:{user_id}")
            if cached_profile:
                return ServiceResult.success_result(cached_profile)
            
            async with self.uow:
                user = await self.uow.users.find_by_id(UserId(user_id))
                if not user:
                    return ServiceResult.error_result("User not found", 404)
                
                profile_data = {
                    "user_id": str(user.id),
                    "email": str(user.email),
                    "role": user.role,
                    "is_active": user.is_active,
                    "is_verified": user.is_verified,
                    "two_factor_enabled": user.two_factor_enabled,
                    "created_at": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None
                }
                
                # Cache profile data
                await self.cache_service.set(f"user_profile:{user_id}", profile_data, ttl=1800)
                
                return ServiceResult.success_result(profile_data)
        
        except Exception as e:
            self.logger.error(f"Failed to get user profile", error=str(e))
            return ServiceResult.error_result("Failed to get profile", 500)
    
    async def _publish_domain_events(self):
        """Publish domain events to event bus."""
        try:
            async with self.uow:
                unpublished_events = await self.uow.events.find_unpublished_events()
                
                for event in unpublished_events:
                    await self.event_publisher.publish(event)
                    await self.uow.events.mark_event_as_published(event.event_id)
                
                await self.uow.commit()
        
        except Exception as e:
            self.logger.error(f"Failed to publish domain events", error=str(e))


class FileProcessingService:
    """Application service for file processing."""
    
    def __init__(
        self,
        uow: UnitOfWork,
        file_storage: FileStorageService,
        task_queue: TaskQueueService,
        event_publisher: EventPublisher,
        cache_service: CacheService
    ):
        self.uow = uow
        self.file_storage = file_storage
        self.task_queue = task_queue
        self.event_publisher = event_publisher
        self.cache_service = cache_service
        self.logger = get_logger(__name__)
    
    async def upload_file(self, user_id: str, file_name: str, file_content: bytes) -> ServiceResult:
        """Upload and process a file."""
        try:
            # Save file to storage
            file_path = f"uploads/{user_id}/{datetime.now().timestamp()}_{file_name}"
            if not await self.file_storage.save_file(file_path, file_content):
                return ServiceResult.error_result("Failed to save file", 500)
            
            # Create upload record
            use_case = UploadFileUseCase(self.uow)
            result = await use_case.execute(UploadFileUseCase.UploadFileRequest(
                user_id=user_id,
                file_name=file_name,
                file_content=file_content,
                file_size=len(file_content)
            ))
            
            if result.success:
                # Enqueue processing task
                task_id = await self.task_queue.enqueue_task(
                    "process_file",
                    {
                        "upload_id": result.data["upload_id"],
                        "user_id": user_id,
                        "file_path": file_path
                    }
                )
                
                # Publish events
                await self._publish_domain_events()
                
                return ServiceResult.success_result({
                    **result.data,
                    "task_id": task_id
                }, 201)
            else:
                # Clean up file if upload failed
                await self.file_storage.delete_file(file_path)
                return ServiceResult.error_result(result.error_message, 400)
        
        except Exception as e:
            self.logger.error(f"File upload failed", error=str(e))
            return ServiceResult.error_result("Upload failed", 500)
    
    async def get_upload_status(self, upload_id: str, user_id: str) -> ServiceResult:
        """Get upload processing status."""
        try:
            # Try cache first
            cached_status = await self.cache_service.get(f"upload_status:{upload_id}")
            if cached_status:
                return ServiceResult.success_result(cached_status)
            
            async with self.uow:
                upload = await self.uow.file_uploads.find_by_id(UploadId(upload_id))
                if not upload:
                    return ServiceResult.error_result("Upload not found", 404)
                
                # Check permission
                if str(upload.uploader_id) != user_id:
                    user = await self.uow.users.find_by_id(UserId(user_id))
                    if not user or not user.has_permission("view_all_uploads"):
                        return ServiceResult.error_result("Access denied", 403)
                
                status_data = {
                    "upload_id": str(upload.id),
                    "file_name": str(upload.file_name),
                    "status": str(upload.status),
                    "progress_percentage": upload.progress_percentage,
                    "total_reviews": upload.total_reviews,
                    "processed_reviews": upload.processed_reviews,
                    "failed_reviews": upload.failed_reviews,
                    "error_message": upload.error_message,
                    "uploaded_at": upload.uploaded_at.isoformat(),
                    "processing_started_at": upload.processing_started_at.isoformat() if upload.processing_started_at else None,
                    "processing_completed_at": upload.processing_completed_at.isoformat() if upload.processing_completed_at else None
                }
                
                # Cache status for short time
                await self.cache_service.set(f"upload_status:{upload_id}", status_data, ttl=30)
                
                return ServiceResult.success_result(status_data)
        
        except Exception as e:
            self.logger.error(f"Failed to get upload status", error=str(e))
            return ServiceResult.error_result("Failed to get status", 500)
    
    async def cancel_upload(self, upload_id: str, user_id: str) -> ServiceResult:
        """Cancel an upload."""
        try:
            async with self.uow:
                upload = await self.uow.file_uploads.find_by_id(UploadId(upload_id))
                if not upload:
                    return ServiceResult.error_result("Upload not found", 404)
                
                # Check permission
                if str(upload.uploader_id) != user_id:
                    user = await self.uow.users.find_by_id(UserId(user_id))
                    if not user or not user.has_permission("cancel_uploads"):
                        return ServiceResult.error_result("Access denied", 403)
                
                # Can only cancel pending or in-progress uploads
                if upload.status not in [ProcessingStatus.PENDING, ProcessingStatus.IN_PROGRESS]:
                    return ServiceResult.error_result("Cannot cancel upload in current status", 400)
                
                # Update status
                upload.status = ProcessingStatus.CANCELLED
                await self.uow.file_uploads.save(upload)
                
                # Clear cache
                await self.cache_service.delete(f"upload_status:{upload_id}")
                
                await self.uow.commit()
                
                return ServiceResult.success_result({"message": "Upload cancelled successfully"})
        
        except Exception as e:
            self.logger.error(f"Failed to cancel upload", error=str(e))
            return ServiceResult.error_result("Failed to cancel upload", 500)
    
    async def _publish_domain_events(self):
        """Publish domain events to event bus."""
        try:
            async with self.uow:
                unpublished_events = await self.uow.events.find_unpublished_events()
                
                for event in unpublished_events:
                    await self.event_publisher.publish(event)
                    await self.uow.events.mark_event_as_published(event.event_id)
                
                await self.uow.commit()
        
        except Exception as e:
            self.logger.error(f"Failed to publish domain events", error=str(e))


class AnalyticsService:
    """Application service for analytics."""
    
    def __init__(
        self,
        uow: UnitOfWork,
        analytics_repo: AnalyticsQueryRepository,
        cache_service: CacheService
    ):
        self.uow = uow
        self.analytics_repo = analytics_repo
        self.cache_service = cache_service
        self.logger = get_logger(__name__)
    
    async def get_sentiment_analysis(self, upload_id: str, user_id: str) -> ServiceResult:
        """Get sentiment analysis results."""
        try:
            use_case = GetSentimentAnalysisUseCase(self.uow)
            result = await use_case.execute(GetSentimentAnalysisUseCase.GetSentimentAnalysisRequest(
                upload_id=upload_id,
                user_id=user_id
            ))
            
            if result.success:
                return ServiceResult.success_result(result.data)
            else:
                return ServiceResult.error_result(result.error_message, 400)
        
        except Exception as e:
            self.logger.error(f"Failed to get sentiment analysis", error=str(e))
            return ServiceResult.error_result("Failed to get analysis", 500)
    
    async def get_user_uploads(self, user_id: str, limit: int = 50) -> ServiceResult:
        """Get user uploads."""
        try:
            # Try cache first
            cache_key = f"user_uploads:{user_id}:{limit}"
            cached_uploads = await self.cache_service.get(cache_key)
            if cached_uploads:
                return ServiceResult.success_result(cached_uploads)
            
            use_case = GetUserUploadsUseCase(self.uow)
            result = await use_case.execute(GetUserUploadsUseCase.GetUserUploadsRequest(
                user_id=user_id,
                limit=limit
            ))
            
            if result.success:
                # Cache for 5 minutes
                await self.cache_service.set(cache_key, result.data, ttl=300)
                return ServiceResult.success_result(result.data)
            else:
                return ServiceResult.error_result(result.error_message, 400)
        
        except Exception as e:
            self.logger.error(f"Failed to get user uploads", error=str(e))
            return ServiceResult.error_result("Failed to get uploads", 500)
    
    async def get_analytics_dashboard(self, user_id: str, days: int = 30) -> ServiceResult:
        """Get analytics dashboard data."""
        try:
            # Try cache first
            cache_key = f"analytics_dashboard:{user_id}:{days}"
            cached_analytics = await self.cache_service.get(cache_key)
            if cached_analytics:
                return ServiceResult.success_result(cached_analytics)
            
            use_case = GetAnalyticsUseCase(self.uow, self.analytics_repo)
            result = await use_case.execute(GetAnalyticsUseCase.GetAnalyticsRequest(
                user_id=user_id,
                days=days
            ))
            
            if result.success:
                # Cache for 10 minutes
                await self.cache_service.set(cache_key, result.data, ttl=600)
                return ServiceResult.success_result(result.data)
            else:
                return ServiceResult.error_result(result.error_message, 400)
        
        except Exception as e:
            self.logger.error(f"Failed to get analytics", error=str(e))
            return ServiceResult.error_result("Failed to get analytics", 500)
    
    async def export_analysis_results(self, upload_id: str, user_id: str, format: str = "csv") -> ServiceResult:
        """Export analysis results."""
        try:
            # Get sentiment analysis
            sentiment_result = await self.get_sentiment_analysis(upload_id, user_id)
            if not sentiment_result.success:
                return sentiment_result
            
            # Generate export data based on format
            if format.lower() == "csv":
                export_data = self._generate_csv_export(sentiment_result.data)
            elif format.lower() == "json":
                export_data = sentiment_result.data
            else:
                return ServiceResult.error_result("Unsupported export format", 400)
            
            return ServiceResult.success_result({
                "upload_id": upload_id,
                "format": format,
                "data": export_data,
                "exported_at": datetime.utcnow().isoformat()
            })
        
        except Exception as e:
            self.logger.error(f"Failed to export analysis results", error=str(e))
            return ServiceResult.error_result("Failed to export results", 500)
    
    def _generate_csv_export(self, analysis_data: Dict[str, Any]) -> str:
        """Generate CSV export data."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow([
            "Review ID", "Sentiment Score", "Sentiment Label", 
            "Confidence", "Provider", "Analyzed At"
        ])
        
        # Write data rows
        for analysis in analysis_data.get("analyses", []):
            writer.writerow([
                analysis["review_id"],
                analysis["sentiment_score"],
                analysis["sentiment_label"],
                analysis["confidence"],
                analysis["provider"],
                analysis["analyzed_at"]
            ])
        
        return output.getvalue()


class SystemAdminService:
    """Application service for system administration."""
    
    def __init__(
        self,
        uow: UnitOfWork,
        analytics_repo: AnalyticsQueryRepository,
        cache_service: CacheService
    ):
        self.uow = uow
        self.analytics_repo = analytics_repo
        self.cache_service = cache_service
        self.logger = get_logger(__name__)
    
    async def get_system_health(self, user_id: str) -> ServiceResult:
        """Get system health status."""
        try:
            use_case = GetSystemHealthUseCase(self.uow)
            result = await use_case.execute(GetSystemHealthUseCase.GetSystemHealthRequest(
                user_id=user_id
            ))
            
            if result.success:
                return ServiceResult.success_result(result.data)
            else:
                return ServiceResult.error_result(result.error_message, 403)
        
        except Exception as e:
            self.logger.error(f"Failed to get system health", error=str(e))
            return ServiceResult.error_result("Failed to get system health", 500)
    
    async def get_system_metrics(self, user_id: str) -> ServiceResult:
        """Get system metrics."""
        try:
            # Check user permission
            async with self.uow:
                user = await self.uow.users.find_by_id(UserId(user_id))
                if not user or not user.has_permission("system_admin"):
                    return ServiceResult.error_result("Access denied", 403)
            
            # Get metrics from analytics repository
            processing_metrics = await self.analytics_repo.get_processing_performance_metrics()
            capacity_data = await self.analytics_repo.get_capacity_planning_data()
            error_analysis = await self.analytics_repo.get_error_analysis()
            
            return ServiceResult.success_result({
                "processing_metrics": processing_metrics,
                "capacity_data": capacity_data,
                "error_analysis": error_analysis,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        except Exception as e:
            self.logger.error(f"Failed to get system metrics", error=str(e))
            return ServiceResult.error_result("Failed to get metrics", 500)
    
    async def cleanup_old_data(self, user_id: str, days_old: int = 90) -> ServiceResult:
        """Clean up old data."""
        try:
            # Check user permission
            async with self.uow:
                user = await self.uow.users.find_by_id(UserId(user_id))
                if not user or not user.has_permission("system_admin"):
                    return ServiceResult.error_result("Access denied", 403)
                
                # Clean up old tasks
                cleaned_tasks = await self.uow.analysis_tasks.cleanup_old_tasks(days_old)
                
                # Clean up old events
                cutoff_date = datetime.utcnow() - timedelta(days=days_old)
                old_events = await self.uow.events.find_events_by_date_range(
                    datetime.min, cutoff_date
                )
                
                for event in old_events:
                    await self.uow.events.delete(event)
                
                await self.uow.commit()
                
                return ServiceResult.success_result({
                    "cleaned_tasks": cleaned_tasks,
                    "cleaned_events": len(old_events),
                    "cutoff_date": cutoff_date.isoformat()
                })
        
        except Exception as e:
            await self.uow.rollback()
            self.logger.error(f"Failed to cleanup old data", error=str(e))
            return ServiceResult.error_result("Failed to cleanup data", 500)


class NotificationService:
    """Application service for notifications."""
    
    def __init__(
        self,
        email_service: EmailNotificationService,
        event_publisher: EventPublisher
    ):
        self.email_service = email_service
        self.event_publisher = event_publisher
        self.logger = get_logger(__name__)
    
    async def send_processing_completed_notification(self, upload: FileUpload, user: User) -> bool:
        """Send notification when file processing is completed."""
        try:
            subject = "Insight Miner: Analysis Completed"
            body = f"""
            Hello {user.email},
            
            Your sentiment analysis for file "{upload.file_name}" has been completed successfully.
            
            Results:
            - Total reviews processed: {upload.processed_reviews}
            - Processing time: {upload.processing_time}
            - Success rate: {(upload.processed_reviews / upload.total_reviews * 100):.1f}%
            
            You can view the results in your dashboard.
            
            Best regards,
            Insight Miner Team
            """
            
            return await self.email_service.send_email(
                str(user.email),
                subject,
                body
            )
        
        except Exception as e:
            self.logger.error(f"Failed to send completion notification", error=str(e))
            return False
    
    async def send_processing_failed_notification(self, upload: FileUpload, user: User) -> bool:
        """Send notification when file processing fails."""
        try:
            subject = "Insight Miner: Analysis Failed"
            body = f"""
            Hello {user.email},
            
            Unfortunately, your sentiment analysis for file "{upload.file_name}" has failed.
            
            Error: {upload.error_message}
            
            Please contact support if you need assistance.
            
            Best regards,
            Insight Miner Team
            """
            
            return await self.email_service.send_email(
                str(user.email),
                subject,
                body
            )
        
        except Exception as e:
            self.logger.error(f"Failed to send failure notification", error=str(e))
            return False