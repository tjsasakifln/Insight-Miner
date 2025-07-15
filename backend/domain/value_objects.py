"""
Value objects for type safety and business rules validation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union
from enum import Enum
import re
from decimal import Decimal
import hashlib


class ValueObject(ABC):
    """Base class for value objects."""
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__
    
    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))


@dataclass(frozen=True)
class UserId(ValueObject):
    """User identifier value object."""
    
    value: str
    
    def __post_init__(self):
        if not self.value:
            raise ValueError("User ID cannot be empty")
        
        if not isinstance(self.value, str):
            raise ValueError("User ID must be a string")
        
        if len(self.value) > 50:
            raise ValueError("User ID cannot exceed 50 characters")
    
    def __str__(self):
        return self.value


@dataclass(frozen=True)
class Email(ValueObject):
    """Email address value object with validation."""
    
    value: str
    
    def __post_init__(self):
        if not self.value:
            raise ValueError("Email cannot be empty")
        
        if not isinstance(self.value, str):
            raise ValueError("Email must be a string")
        
        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, self.value):
            raise ValueError("Invalid email format")
        
        if len(self.value) > 320:  # RFC 5321 limit
            raise ValueError("Email cannot exceed 320 characters")
    
    @property
    def domain(self) -> str:
        """Get email domain."""
        return self.value.split('@')[1]
    
    @property
    def local_part(self) -> str:
        """Get email local part."""
        return self.value.split('@')[0]
    
    def __str__(self):
        return self.value


@dataclass(frozen=True)
class SentimentScore(ValueObject):
    """Sentiment score value object with validation."""
    
    value: float
    
    def __post_init__(self):
        if not isinstance(self.value, (int, float)):
            raise ValueError("Sentiment score must be a number")
        
        if not -1.0 <= self.value <= 1.0:
            raise ValueError("Sentiment score must be between -1.0 and 1.0")
    
    @property
    def is_positive(self) -> bool:
        """Check if sentiment is positive."""
        return self.value > 0.1
    
    @property
    def is_negative(self) -> bool:
        """Check if sentiment is negative."""
        return self.value < -0.1
    
    @property
    def is_neutral(self) -> bool:
        """Check if sentiment is neutral."""
        return -0.1 <= self.value <= 0.1
    
    @property
    def intensity(self) -> str:
        """Get sentiment intensity."""
        abs_value = abs(self.value)
        if abs_value >= 0.8:
            return "very_strong"
        elif abs_value >= 0.6:
            return "strong"
        elif abs_value >= 0.3:
            return "moderate"
        elif abs_value >= 0.1:
            return "weak"
        else:
            return "neutral"
    
    def __str__(self):
        return f"{self.value:.3f}"


@dataclass(frozen=True)
class ReviewText(ValueObject):
    """Review text value object with validation."""
    
    value: str
    
    def __post_init__(self):
        if not isinstance(self.value, str):
            raise ValueError("Review text must be a string")
        
        # Allow empty strings but validate length
        if len(self.value) > 10000:
            raise ValueError("Review text cannot exceed 10,000 characters")
        
        # Remove excessive whitespace
        cleaned_value = ' '.join(self.value.split())
        object.__setattr__(self, 'value', cleaned_value)
    
    @property
    def word_count(self) -> int:
        """Get word count."""
        if not self.value:
            return 0
        return len(self.value.split())
    
    @property
    def character_count(self) -> int:
        """Get character count."""
        return len(self.value)
    
    @property
    def is_empty(self) -> bool:
        """Check if text is empty."""
        return not self.value.strip()
    
    @property
    def hash(self) -> str:
        """Get hash of the text for deduplication."""
        return hashlib.md5(self.value.encode()).hexdigest()
    
    def contains_keyword(self, keyword: str) -> bool:
        """Check if text contains keyword (case-insensitive)."""
        return keyword.lower() in self.value.lower()
    
    def __str__(self):
        return self.value


@dataclass(frozen=True)
class FileName(ValueObject):
    """File name value object with validation."""
    
    value: str
    
    def __post_init__(self):
        if not self.value:
            raise ValueError("File name cannot be empty")
        
        if not isinstance(self.value, str):
            raise ValueError("File name must be a string")
        
        if len(self.value) > 255:
            raise ValueError("File name cannot exceed 255 characters")
        
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        if any(char in self.value for char in invalid_chars):
            raise ValueError("File name contains invalid characters")
    
    @property
    def extension(self) -> str:
        """Get file extension."""
        return self.value.split('.')[-1].lower() if '.' in self.value else ''
    
    @property
    def name_without_extension(self) -> str:
        """Get file name without extension."""
        return '.'.join(self.value.split('.')[:-1]) if '.' in self.value else self.value
    
    @property
    def is_csv(self) -> bool:
        """Check if file is CSV."""
        return self.extension == 'csv'
    
    @property
    def is_excel(self) -> bool:
        """Check if file is Excel."""
        return self.extension in ['xlsx', 'xls']
    
    def __str__(self):
        return self.value


class ProcessingStatus(Enum):
    """Processing status enumeration."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
    def __str__(self):
        return self.value
    
    @property
    def is_terminal(self) -> bool:
        """Check if status is terminal (final)."""
        return self in [self.COMPLETED, self.FAILED, self.CANCELLED]
    
    @property
    def is_active(self) -> bool:
        """Check if status is active (can be updated)."""
        return self in [self.PENDING, self.IN_PROGRESS]


class AnalysisType(Enum):
    """Analysis type enumeration."""
    
    SENTIMENT = "sentiment"
    TOPIC_EXTRACTION = "topic_extraction"
    KEYWORD_EXTRACTION = "keyword_extraction"
    EMOTION_ANALYSIS = "emotion_analysis"
    LANGUAGE_DETECTION = "language_detection"
    
    def __str__(self):
        return self.value
    
    @property
    def requires_ml(self) -> bool:
        """Check if analysis type requires ML processing."""
        return self in [self.SENTIMENT, self.TOPIC_EXTRACTION, self.EMOTION_ANALYSIS]


@dataclass(frozen=True)
class UploadId(ValueObject):
    """Upload identifier value object."""
    
    value: str
    
    def __post_init__(self):
        if not self.value:
            raise ValueError("Upload ID cannot be empty")
        
        if not isinstance(self.value, str):
            raise ValueError("Upload ID must be a string")
        
        if len(self.value) > 50:
            raise ValueError("Upload ID cannot exceed 50 characters")
    
    def __str__(self):
        return self.value


@dataclass(frozen=True)
class TaskId(ValueObject):
    """Task identifier value object."""
    
    value: str
    
    def __post_init__(self):
        if not self.value:
            raise ValueError("Task ID cannot be empty")
        
        if not isinstance(self.value, str):
            raise ValueError("Task ID must be a string")
        
        if len(self.value) > 50:
            raise ValueError("Task ID cannot exceed 50 characters")
    
    def __str__(self):
        return self.value


@dataclass(frozen=True)
class FileSize(ValueObject):
    """File size value object with validation."""
    
    value: int  # Size in bytes
    
    def __post_init__(self):
        if not isinstance(self.value, int):
            raise ValueError("File size must be an integer")
        
        if self.value < 0:
            raise ValueError("File size cannot be negative")
        
        if self.value > 1073741824:  # 1GB limit
            raise ValueError("File size cannot exceed 1GB")
    
    @property
    def kb(self) -> float:
        """Get size in kilobytes."""
        return self.value / 1024
    
    @property
    def mb(self) -> float:
        """Get size in megabytes."""
        return self.value / (1024 * 1024)
    
    @property
    def gb(self) -> float:
        """Get size in gigabytes."""
        return self.value / (1024 * 1024 * 1024)
    
    @property
    def human_readable(self) -> str:
        """Get human-readable size."""
        if self.value < 1024:
            return f"{self.value} B"
        elif self.value < 1024 * 1024:
            return f"{self.kb:.1f} KB"
        elif self.value < 1024 * 1024 * 1024:
            return f"{self.mb:.1f} MB"
        else:
            return f"{self.gb:.1f} GB"
    
    def __str__(self):
        return self.human_readable


@dataclass(frozen=True)
class ProcessingTime(ValueObject):
    """Processing time value object."""
    
    value: float  # Time in seconds
    
    def __post_init__(self):
        if not isinstance(self.value, (int, float)):
            raise ValueError("Processing time must be a number")
        
        if self.value < 0:
            raise ValueError("Processing time cannot be negative")
        
        if self.value > 86400:  # 24 hours limit
            raise ValueError("Processing time cannot exceed 24 hours")
    
    @property
    def milliseconds(self) -> int:
        """Get time in milliseconds."""
        return int(self.value * 1000)
    
    @property
    def minutes(self) -> float:
        """Get time in minutes."""
        return self.value / 60
    
    @property
    def hours(self) -> float:
        """Get time in hours."""
        return self.value / 3600
    
    @property
    def human_readable(self) -> str:
        """Get human-readable time."""
        if self.value < 1:
            return f"{self.milliseconds} ms"
        elif self.value < 60:
            return f"{self.value:.1f} s"
        elif self.value < 3600:
            return f"{self.minutes:.1f} min"
        else:
            return f"{self.hours:.1f} h"
    
    def __str__(self):
        return self.human_readable


@dataclass(frozen=True)
class Confidence(ValueObject):
    """Confidence score value object."""
    
    value: float
    
    def __post_init__(self):
        if not isinstance(self.value, (int, float)):
            raise ValueError("Confidence must be a number")
        
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    @property
    def percentage(self) -> float:
        """Get confidence as percentage."""
        return self.value * 100
    
    @property
    def level(self) -> str:
        """Get confidence level."""
        if self.value >= 0.9:
            return "very_high"
        elif self.value >= 0.7:
            return "high"
        elif self.value >= 0.5:
            return "medium"
        elif self.value >= 0.3:
            return "low"
        else:
            return "very_low"
    
    def __str__(self):
        return f"{self.percentage:.1f}%"


@dataclass(frozen=True)
class Priority(ValueObject):
    """Priority value object."""
    
    value: int
    
    def __post_init__(self):
        if not isinstance(self.value, int):
            raise ValueError("Priority must be an integer")
        
        if not 1 <= self.value <= 10:
            raise ValueError("Priority must be between 1 and 10")
    
    @property
    def level(self) -> str:
        """Get priority level."""
        if self.value >= 8:
            return "critical"
        elif self.value >= 6:
            return "high"
        elif self.value >= 4:
            return "medium"
        elif self.value >= 2:
            return "low"
        else:
            return "very_low"
    
    @property
    def is_urgent(self) -> bool:
        """Check if priority is urgent."""
        return self.value >= 7
    
    def __str__(self):
        return f"{self.value} ({self.level})"


@dataclass(frozen=True)
class RetryCount(ValueObject):
    """Retry count value object."""
    
    value: int
    
    def __post_init__(self):
        if not isinstance(self.value, int):
            raise ValueError("Retry count must be an integer")
        
        if self.value < 0:
            raise ValueError("Retry count cannot be negative")
        
        if self.value > 10:
            raise ValueError("Retry count cannot exceed 10")
    
    @property
    def has_retries_left(self) -> bool:
        """Check if there are retries left."""
        return self.value > 0
    
    def decrement(self) -> 'RetryCount':
        """Decrement retry count."""
        if self.value <= 0:
            raise ValueError("Cannot decrement retry count below 0")
        return RetryCount(self.value - 1)
    
    def __str__(self):
        return str(self.value)