"""
Domain services for complex business logic that doesn't belong to entities.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import statistics
from decimal import Decimal

from .entities import User, FileUpload, SentimentAnalysis, ReviewData
from .value_objects import (
    UserId, UploadId, SentimentScore, ReviewText, 
    ProcessingStatus, AnalysisType, Confidence, Priority
)
from .events import (
    FileUploadedEvent, BatchAnalysisCompletedEvent,
    SystemMetricsEvent, DomainEvent
)


class ValidationResult:
    """Result of validation operations."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result."""
        self.errors.extend(other.errors)
        self.is_valid = self.is_valid and other.is_valid


@dataclass
class AnalysisMetrics:
    """Metrics for analysis operations."""
    
    total_reviews: int
    processed_reviews: int
    failed_reviews: int
    average_processing_time: float
    sentiment_distribution: Dict[str, int]
    confidence_distribution: Dict[str, int]
    provider_performance: Dict[str, Dict[str, float]]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_reviews == 0:
            return 0.0
        return (self.processed_reviews / self.total_reviews) * 100
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_reviews == 0:
            return 0.0
        return (self.failed_reviews / self.total_reviews) * 100
    
    @property
    def throughput(self) -> float:
        """Calculate throughput (reviews per second)."""
        if self.average_processing_time == 0:
            return 0.0
        return 1.0 / self.average_processing_time


class FileValidationService:
    """Service for validating uploaded files."""
    
    def __init__(self, max_file_size: int = 100 * 1024 * 1024):  # 100MB
        self.max_file_size = max_file_size
        self.allowed_extensions = ['.csv', '.xlsx', '.xls', '.json']
        self.required_columns = ['review_text']
        self.optional_columns = ['rating', 'date', 'product_id', 'user_id']
    
    def validate_file_upload(self, file_upload: FileUpload) -> ValidationResult:
        """Validate a file upload."""
        result = ValidationResult(True)
        
        # Validate file size
        if not file_upload.validate_file_size(self.max_file_size):
            result.add_error(f"File size exceeds maximum allowed size of {self.max_file_size} bytes")
        
        # Validate file type
        if not file_upload.validate_file_type(self.allowed_extensions):
            result.add_error(f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}")
        
        # Validate file name
        if not file_upload.file_name.value:
            result.add_error("File name cannot be empty")
        
        return result
    
    def validate_file_content(self, reviews: List[ReviewData]) -> ValidationResult:
        """Validate file content."""
        result = ValidationResult(True)
        
        if not reviews:
            result.add_error("File contains no reviews")
            return result
        
        # Check for required columns (assuming reviews have been parsed)
        for i, review in enumerate(reviews):
            if not review.text.value.strip():
                result.add_error(f"Review {i + 1} has empty text")
            
            if review.text.character_count > 10000:
                result.add_error(f"Review {i + 1} exceeds maximum character limit")
        
        # Check for duplicates
        text_hashes = [review.text.hash for review in reviews]
        unique_hashes = set(text_hashes)
        
        if len(unique_hashes) != len(text_hashes):
            duplicates = len(text_hashes) - len(unique_hashes)
            result.add_error(f"Found {duplicates} duplicate reviews")
        
        return result
    
    def estimate_processing_time(self, review_count: int) -> timedelta:
        """Estimate processing time based on review count."""
        # Assume 0.5 seconds per review on average
        seconds = review_count * 0.5
        return timedelta(seconds=seconds)


class SentimentAnalysisService:
    """Service for sentiment analysis business logic."""
    
    def __init__(self):
        self.sentiment_thresholds = {
            'very_positive': 0.6,
            'positive': 0.1,
            'neutral': 0.1,
            'negative': -0.1,
            'very_negative': -0.6
        }
    
    def categorize_sentiment(self, sentiment_score: SentimentScore) -> str:
        """Categorize sentiment score into labels."""
        score = sentiment_score.value
        
        if score >= self.sentiment_thresholds['very_positive']:
            return 'very_positive'
        elif score >= self.sentiment_thresholds['positive']:
            return 'positive'
        elif score >= self.sentiment_thresholds['neutral']:
            return 'neutral'
        elif score >= self.sentiment_thresholds['negative']:
            return 'negative'
        else:
            return 'very_negative'
    
    def calculate_sentiment_distribution(self, analyses: List[SentimentAnalysis]) -> Dict[str, int]:
        """Calculate sentiment distribution."""
        distribution = {
            'very_positive': 0,
            'positive': 0,
            'neutral': 0,
            'negative': 0,
            'very_negative': 0
        }
        
        for analysis in analyses:
            category = self.categorize_sentiment(analysis.sentiment_score)
            distribution[category] += 1
        
        return distribution
    
    def calculate_average_sentiment(self, analyses: List[SentimentAnalysis]) -> Optional[SentimentScore]:
        """Calculate average sentiment score."""
        if not analyses:
            return None
        
        total_score = sum(analysis.sentiment_score.value for analysis in analyses)
        average_score = total_score / len(analyses)
        
        return SentimentScore(average_score)
    
    def identify_sentiment_trends(self, analyses: List[SentimentAnalysis]) -> Dict[str, Any]:
        """Identify sentiment trends over time."""
        if not analyses:
            return {}
        
        # Sort analyses by date
        sorted_analyses = sorted(analyses, key=lambda x: x.analyzed_at)
        
        # Calculate rolling averages
        window_size = min(100, len(sorted_analyses) // 10)  # 10% of data or 100, whichever is smaller
        rolling_averages = []
        
        for i in range(window_size, len(sorted_analyses)):
            window = sorted_analyses[i-window_size:i]
            avg_score = sum(a.sentiment_score.value for a in window) / len(window)
            rolling_averages.append({
                'timestamp': sorted_analyses[i].analyzed_at,
                'average_sentiment': avg_score
            })
        
        # Calculate trend direction
        if len(rolling_averages) >= 2:
            first_half = rolling_averages[:len(rolling_averages)//2]
            second_half = rolling_averages[len(rolling_averages)//2:]
            
            first_avg = sum(item['average_sentiment'] for item in first_half) / len(first_half)
            second_avg = sum(item['average_sentiment'] for item in second_half) / len(second_half)
            
            trend = 'improving' if second_avg > first_avg else 'declining' if second_avg < first_avg else 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'rolling_averages': rolling_averages,
            'total_analyses': len(analyses)
        }
    
    def detect_sentiment_anomalies(self, analyses: List[SentimentAnalysis]) -> List[Dict[str, Any]]:
        """Detect anomalies in sentiment scores."""
        if len(analyses) < 10:
            return []
        
        scores = [analysis.sentiment_score.value for analysis in analyses]
        mean_score = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
        
        anomalies = []
        threshold = 2.0  # 2 standard deviations
        
        for analysis in analyses:
            score = analysis.sentiment_score.value
            z_score = abs(score - mean_score) / std_dev if std_dev > 0 else 0
            
            if z_score > threshold:
                anomalies.append({
                    'review_id': analysis.review_id,
                    'sentiment_score': score,
                    'z_score': z_score,
                    'review_text': analysis.review_text.value[:100] + '...',
                    'analyzed_at': analysis.analyzed_at
                })
        
        return anomalies


class UserPermissionService:
    """Service for user permission and authorization logic."""
    
    def __init__(self):
        self.role_hierarchy = {
            'viewer': 1,
            'analyst': 2,
            'admin': 3,
            'super_admin': 4
        }
        
        self.permissions = {
            'upload_files': ['analyst', 'admin', 'super_admin'],
            'view_analytics': ['viewer', 'analyst', 'admin', 'super_admin'],
            'manage_users': ['admin', 'super_admin'],
            'system_admin': ['super_admin'],
            'export_data': ['analyst', 'admin', 'super_admin'],
            'delete_uploads': ['admin', 'super_admin']
        }
    
    def can_user_perform_action(self, user: User, action: str) -> bool:
        """Check if user can perform specific action."""
        if not user.is_active:
            return False
        
        allowed_roles = self.permissions.get(action, [])
        return user.role in allowed_roles
    
    def can_user_access_upload(self, user: User, upload: FileUpload) -> bool:
        """Check if user can access specific upload."""
        if not user.is_active:
            return False
        
        # Users can always access their own uploads
        if upload.uploader_id == user.id:
            return True
        
        # Admins can access all uploads
        if user.role in ['admin', 'super_admin']:
            return True
        
        return False
    
    def get_user_access_level(self, user: User) -> int:
        """Get user access level based on role."""
        return self.role_hierarchy.get(user.role, 0)
    
    def can_user_elevate_role(self, acting_user: User, target_role: str) -> bool:
        """Check if user can elevate someone to target role."""
        if not acting_user.is_active:
            return False
        
        acting_level = self.get_user_access_level(acting_user)
        target_level = self.role_hierarchy.get(target_role, 0)
        
        # User must have higher level than target role
        return acting_level > target_level


class MetricsCalculationService:
    """Service for calculating business metrics."""
    
    def calculate_processing_metrics(self, uploads: List[FileUpload]) -> Dict[str, Any]:
        """Calculate processing metrics for uploads."""
        if not uploads:
            return {}
        
        total_uploads = len(uploads)
        completed_uploads = len([u for u in uploads if u.is_completed])
        failed_uploads = len([u for u in uploads if u.is_failed])
        in_progress_uploads = len([u for u in uploads if u.is_in_progress])
        
        # Calculate processing times for completed uploads
        processing_times = []
        for upload in uploads:
            if upload.is_completed and upload.processing_time:
                processing_times.append(upload.processing_time.total_seconds())
        
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0
        
        # Calculate throughput (uploads per hour)
        if uploads:
            time_span = max(u.uploaded_at for u in uploads) - min(u.uploaded_at for u in uploads)
            hours = time_span.total_seconds() / 3600
            throughput = total_uploads / hours if hours > 0 else 0
        else:
            throughput = 0
        
        return {
            'total_uploads': total_uploads,
            'completed_uploads': completed_uploads,
            'failed_uploads': failed_uploads,
            'in_progress_uploads': in_progress_uploads,
            'success_rate': (completed_uploads / total_uploads) * 100 if total_uploads > 0 else 0,
            'average_processing_time': avg_processing_time,
            'throughput_per_hour': throughput
        }
    
    def calculate_sentiment_metrics(self, analyses: List[SentimentAnalysis]) -> AnalysisMetrics:
        """Calculate comprehensive sentiment analysis metrics."""
        if not analyses:
            return AnalysisMetrics(
                total_reviews=0,
                processed_reviews=0,
                failed_reviews=0,
                average_processing_time=0,
                sentiment_distribution={},
                confidence_distribution={},
                provider_performance={}
            )
        
        # Basic counts
        total_reviews = len(analyses)
        processed_reviews = total_reviews  # All in list are processed
        failed_reviews = 0  # Would need to track separately
        
        # Processing time
        processing_times = [a.processing_time for a in analyses]
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0
        
        # Sentiment distribution
        sentiment_service = SentimentAnalysisService()
        sentiment_distribution = sentiment_service.calculate_sentiment_distribution(analyses)
        
        # Confidence distribution
        confidence_ranges = {
            'very_high': (0.9, 1.0),
            'high': (0.7, 0.9),
            'medium': (0.5, 0.7),
            'low': (0.3, 0.5),
            'very_low': (0.0, 0.3)
        }
        
        confidence_distribution = {range_name: 0 for range_name in confidence_ranges.keys()}
        
        for analysis in analyses:
            for range_name, (min_val, max_val) in confidence_ranges.items():
                if min_val <= analysis.confidence < max_val:
                    confidence_distribution[range_name] += 1
                    break
        
        # Provider performance
        provider_stats = {}
        for analysis in analyses:
            provider = analysis.provider
            if provider not in provider_stats:
                provider_stats[provider] = {
                    'count': 0,
                    'total_time': 0,
                    'total_confidence': 0
                }
            
            provider_stats[provider]['count'] += 1
            provider_stats[provider]['total_time'] += analysis.processing_time
            provider_stats[provider]['total_confidence'] += analysis.confidence
        
        provider_performance = {}
        for provider, stats in provider_stats.items():
            count = stats['count']
            provider_performance[provider] = {
                'average_processing_time': stats['total_time'] / count,
                'average_confidence': stats['total_confidence'] / count,
                'usage_percentage': (count / total_reviews) * 100
            }
        
        return AnalysisMetrics(
            total_reviews=total_reviews,
            processed_reviews=processed_reviews,
            failed_reviews=failed_reviews,
            average_processing_time=avg_processing_time,
            sentiment_distribution=sentiment_distribution,
            confidence_distribution=confidence_distribution,
            provider_performance=provider_performance
        )


class BusinessRuleService:
    """Service for enforcing business rules."""
    
    def __init__(self):
        self.max_concurrent_uploads_per_user = 3
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.max_reviews_per_file = 100000
        self.upload_cooldown_minutes = 5
    
    def can_user_upload_file(self, user: User, active_uploads: List[FileUpload]) -> Tuple[bool, str]:
        """Check if user can upload a new file."""
        if not user.is_active:
            return False, "User account is not active"
        
        if not user.is_verified:
            return False, "User account is not verified"
        
        # Check concurrent uploads
        user_active_uploads = [u for u in active_uploads if u.uploader_id == user.id and u.is_in_progress]
        if len(user_active_uploads) >= self.max_concurrent_uploads_per_user:
            return False, f"Maximum concurrent uploads ({self.max_concurrent_uploads_per_user}) exceeded"
        
        # Check cooldown period
        user_recent_uploads = [u for u in active_uploads if u.uploader_id == user.id]
        if user_recent_uploads:
            latest_upload = max(user_recent_uploads, key=lambda x: x.uploaded_at)
            time_since_last = datetime.utcnow() - latest_upload.uploaded_at
            if time_since_last < timedelta(minutes=self.upload_cooldown_minutes):
                remaining_time = self.upload_cooldown_minutes - time_since_last.total_seconds() / 60
                return False, f"Upload cooldown active. Wait {remaining_time:.1f} minutes"
        
        return True, ""
    
    def validate_file_for_processing(self, file_upload: FileUpload) -> ValidationResult:
        """Validate file before processing."""
        result = ValidationResult(True)
        
        # Check file size
        if file_upload.file_size > self.max_file_size:
            result.add_error(f"File size exceeds maximum limit of {self.max_file_size} bytes")
        
        # Check review count
        if file_upload.total_reviews > self.max_reviews_per_file:
            result.add_error(f"File contains too many reviews. Maximum: {self.max_reviews_per_file}")
        
        # Check file status
        if file_upload.status != ProcessingStatus.PENDING:
            result.add_error(f"File is not in pending status. Current status: {file_upload.status}")
        
        return result
    
    def calculate_processing_priority(self, user: User, file_upload: FileUpload) -> Priority:
        """Calculate processing priority based on business rules."""
        base_priority = 5  # Default priority
        
        # Higher priority for admin users
        if user.role in ['admin', 'super_admin']:
            base_priority += 2
        
        # Higher priority for smaller files (faster processing)
        if file_upload.file_size < 1024 * 1024:  # Less than 1MB
            base_priority += 1
        
        # Higher priority for premium users (if we had that concept)
        # base_priority += 1 if user.is_premium else 0
        
        return Priority(min(base_priority, 10))  # Cap at 10