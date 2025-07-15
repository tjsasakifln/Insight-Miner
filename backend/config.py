"""
Enterprise-grade configuration management with security best practices.
"""
import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from cryptography.fernet import Fernet
import secrets


class SecurityConfig(BaseSettings):
    """Security configuration with validation and encryption."""
    
    jwt_secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(64),
        description="JWT secret key for token signing"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, ge=1, le=168)
    
    encryption_key: str = Field(
        default_factory=lambda: Fernet.generate_key().decode(),
        description="Encryption key for sensitive data"
    )
    
    password_min_length: int = Field(default=12, ge=8, le=128)
    password_require_special: bool = Field(default=True)
    
    rate_limit_requests: int = Field(default=100, ge=1, le=10000)
    rate_limit_window: int = Field(default=3600, ge=60, le=86400)
    
    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False


class DatabaseConfig(BaseSettings):
    """Database configuration with connection pooling."""
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(default="insight_miner", description="Database name")
    username: str = Field(default="postgres", description="Database username")
    password: str = Field(description="Database password")
    
    pool_size: int = Field(default=20, ge=1, le=100)
    max_overflow: int = Field(default=30, ge=0, le=100)
    pool_timeout: int = Field(default=30, ge=1, le=300)
    pool_recycle: int = Field(default=3600, ge=300, le=86400)
    
    ssl_mode: str = Field(default="prefer", description="SSL mode")
    connect_timeout: int = Field(default=10, ge=1, le=60)
    
    @validator('password')
    def validate_password(cls, v):
        if not v or len(v) < 8:
            raise ValueError('Database password must be at least 8 characters')
        return v
    
    @property
    def connection_string(self) -> str:
        """Generate secure connection string."""
        return (
            f"postgresql+asyncpg://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}&connect_timeout={self.connect_timeout}"
        )
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False


class RedisConfig(BaseSettings):
    """Redis configuration for caching and queuing."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: Optional[str] = Field(default=None, description="Redis password")
    
    max_connections: int = Field(default=50, ge=1, le=1000)
    connection_timeout: int = Field(default=5, ge=1, le=30)
    socket_keepalive: bool = Field(default=True)
    
    ssl_enabled: bool = Field(default=False)
    ssl_cert_reqs: str = Field(default="required")
    
    @property
    def connection_string(self) -> str:
        """Generate Redis connection string."""
        auth = f":{self.password}@" if self.password else ""
        protocol = "rediss" if self.ssl_enabled else "redis"
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_prefix = "REDIS_"
        case_sensitive = False


class ExternalServicesConfig(BaseSettings):
    """External services configuration."""
    
    openai_api_key: str = Field(description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="OpenAI model")
    openai_max_tokens: int = Field(default=2000, ge=1, le=8000)
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    openai_timeout: int = Field(default=30, ge=1, le=300)
    
    google_credentials_path: Optional[str] = Field(
        default=None, description="Google Cloud credentials path"
    )
    google_project_id: Optional[str] = Field(
        default=None, description="Google Cloud project ID"
    )
    
    aws_access_key_id: Optional[str] = Field(
        default=None, description="AWS access key ID"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS secret access key"
    )
    aws_region: str = Field(default="us-east-1", description="AWS region")
    
    @validator('openai_api_key')
    def validate_openai_key(cls, v):
        if not v or not v.startswith('sk-'):
            raise ValueError('Invalid OpenAI API key format')
        return v
    
    class Config:
        env_prefix = "EXTERNAL_"
        case_sensitive = False


class ApplicationConfig(BaseSettings):
    """Main application configuration."""
    
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1, le=32)
    
    cors_origins: list = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        description="CORS origins"
    )
    
    file_upload_max_size: int = Field(default=104857600, ge=1048576, le=1073741824)  # 100MB
    file_upload_allowed_types: list = Field(
        default=[".csv", ".xlsx", ".json"],
        description="Allowed file types"
    )
    
    batch_size: int = Field(default=100, ge=1, le=10000)
    max_concurrent_tasks: int = Field(default=10, ge=1, le=100)
    
    @validator('environment')
    def validate_environment(cls, v):
        if v not in ['development', 'staging', 'production']:
            raise ValueError('Environment must be development, staging, or production')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        if v not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError('Invalid log level')
        return v
    
    class Config:
        env_prefix = "APP_"
        case_sensitive = False


class Settings:
    """Centralized settings management."""
    
    def __init__(self):
        self.security = SecurityConfig()
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.external_services = ExternalServicesConfig()
        self.application = ApplicationConfig()
        self._encryption_key = None
    
    @property
    def encryption_key(self) -> Fernet:
        """Get encryption key for sensitive data."""
        if self._encryption_key is None:
            self._encryption_key = Fernet(self.security.encryption_key.encode())
        return self._encryption_key
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.encryption_key.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.encryption_key.decrypt(encrypted_data.encode()).decode()
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings for debugging (excluding sensitive data)."""
        return {
            'security': {
                'jwt_algorithm': self.security.jwt_algorithm,
                'jwt_expiration_hours': self.security.jwt_expiration_hours,
                'password_min_length': self.security.password_min_length,
                'rate_limit_requests': self.security.rate_limit_requests,
                'rate_limit_window': self.security.rate_limit_window,
            },
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'database': self.database.database,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow,
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'db': self.redis.db,
                'max_connections': self.redis.max_connections,
            },
            'application': {
                'environment': self.application.environment,
                'debug': self.application.debug,
                'log_level': self.application.log_level,
                'host': self.application.host,
                'port': self.application.port,
                'workers': self.application.workers,
            }
        }


# Global settings instance
settings = Settings()