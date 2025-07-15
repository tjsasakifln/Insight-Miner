"""
Enterprise-grade logging system with structured logging, monitoring, and security.
"""
import os
import sys
import json
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from loguru import logger
import structlog
from pythonjsonlogger import jsonlogger
from opentelemetry import trace
from opentelemetry.instrumentation.logging import LoggingInstrumentor

from .config import settings


class SecurityFilter:
    """Filter sensitive information from logs."""
    
    SENSITIVE_PATTERNS = [
        'password', 'secret', 'token', 'key', 'credential',
        'authorization', 'bearer', 'jwt', 'oauth',
        'api_key', 'access_key', 'secret_key'
    ]
    
    @staticmethod
    def filter_sensitive_data(record: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from log records."""
        filtered_record = record.copy()
        
        def filter_dict(data: Dict[str, Any]) -> Dict[str, Any]:
            filtered = {}
            for key, value in data.items():
                key_lower = key.lower()
                if any(pattern in key_lower for pattern in SecurityFilter.SENSITIVE_PATTERNS):
                    filtered[key] = "[REDACTED]"
                elif isinstance(value, dict):
                    filtered[key] = filter_dict(value)
                elif isinstance(value, list):
                    filtered[key] = [filter_dict(item) if isinstance(item, dict) else item for item in value]
                else:
                    filtered[key] = value
            return filtered
        
        if isinstance(filtered_record.get('message'), dict):
            filtered_record['message'] = filter_dict(filtered_record['message'])
        
        return filtered_record


class StructuredLogger:
    """Enterprise-grade structured logging with security and monitoring."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logger.bind(service=name)
        self.tracer = trace.get_tracer(__name__)
        
        # Setup structured logging
        self._setup_structured_logging()
        
        # Setup file handlers
        self._setup_file_handlers()
        
        # Setup monitoring
        self._setup_monitoring()
    
    def _setup_structured_logging(self):
        """Configure structured logging with JSON format."""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Setup JSON formatter
        json_formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Configure loguru
        logger.remove()  # Remove default handler
        
        # Console handler with colors in development
        if settings.application.environment == "development":
            logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=settings.application.log_level,
                colorize=True,
                backtrace=True,
                diagnose=True
            )
        else:
            logger.add(
                sys.stderr,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level=settings.application.log_level,
                serialize=True  # JSON format in production
            )
    
    def _setup_file_handlers(self):
        """Setup file-based logging handlers."""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Application logs
        logger.add(
            logs_dir / "application.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="INFO",
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            serialize=True
        )
        
        # Error logs
        logger.add(
            logs_dir / "errors.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="50 MB",
            retention="90 days",
            compression="gz",
            serialize=True,
            backtrace=True,
            diagnose=True
        )
        
        # Security logs
        logger.add(
            logs_dir / "security.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | SECURITY | {name}:{function}:{line} - {message}",
            level="INFO",
            rotation="100 MB",
            retention="365 days",
            compression="gz",
            serialize=True,
            filter=lambda record: record["extra"].get("security", False)
        )
        
        # Audit logs
        logger.add(
            logs_dir / "audit.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | AUDIT | {name}:{function}:{line} - {message}",
            level="INFO",
            rotation="100 MB",
            retention="2555 days",  # 7 years retention for compliance
            compression="gz",
            serialize=True,
            filter=lambda record: record["extra"].get("audit", False)
        )
    
    def _setup_monitoring(self):
        """Setup monitoring and tracing integration."""
        # Initialize OpenTelemetry logging instrumentation
        LoggingInstrumentor().instrument(set_logging_format=True)
    
    def _get_context(self) -> Dict[str, Any]:
        """Get current execution context."""
        context = {
            "service": self.name,
            "timestamp": datetime.utcnow().isoformat(),
            "environment": settings.application.environment
        }
        
        # Add tracing context if available
        span = trace.get_current_span()
        if span:
            span_context = span.get_span_context()
            context.update({
                "trace_id": format(span_context.trace_id, "032x"),
                "span_id": format(span_context.span_id, "016x")
            })
        
        return context
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        context = self._get_context()
        filtered_context = SecurityFilter.filter_sensitive_data({**context, **kwargs})
        self.logger.info(message, **filtered_context)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        context = self._get_context()
        filtered_context = SecurityFilter.filter_sensitive_data({**context, **kwargs})
        self.logger.debug(message, **filtered_context)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        context = self._get_context()
        filtered_context = SecurityFilter.filter_sensitive_data({**context, **kwargs})
        self.logger.warning(message, **filtered_context)
    
    def error(self, message: str, **kwargs):
        """Log error message with context and stack trace."""
        context = self._get_context()
        
        # Add exception info if available
        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            context["exception"] = {
                "type": exc_info[0].__name__,
                "message": str(exc_info[1]),
                "traceback": traceback.format_exception(*exc_info)
            }
        
        filtered_context = SecurityFilter.filter_sensitive_data({**context, **kwargs})
        self.logger.error(message, **filtered_context)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        context = self._get_context()
        filtered_context = SecurityFilter.filter_sensitive_data({**context, **kwargs})
        self.logger.critical(message, **filtered_context)
    
    def security(self, message: str, **kwargs):
        """Log security-related message."""
        context = self._get_context()
        context["security"] = True
        filtered_context = SecurityFilter.filter_sensitive_data({**context, **kwargs})
        self.logger.info(message, **filtered_context)
    
    def audit(self, message: str, **kwargs):
        """Log audit message for compliance."""
        context = self._get_context()
        context["audit"] = True
        filtered_context = SecurityFilter.filter_sensitive_data({**context, **kwargs})
        self.logger.info(message, **filtered_context)
    
    def performance(self, message: str, duration: float, **kwargs):
        """Log performance metrics."""
        context = self._get_context()
        context.update({
            "performance": True,
            "duration_ms": duration * 1000,
            "duration_seconds": duration
        })
        filtered_context = SecurityFilter.filter_sensitive_data({**context, **kwargs})
        self.logger.info(message, **filtered_context)
    
    def user_action(self, user_id: str, action: str, resource: str, **kwargs):
        """Log user action for audit trail."""
        context = self._get_context()
        context.update({
            "audit": True,
            "user_id": user_id,
            "action": action,
            "resource": resource
        })
        filtered_context = SecurityFilter.filter_sensitive_data({**context, **kwargs})
        self.logger.info(f"User {user_id} performed {action} on {resource}", **filtered_context)
    
    def api_request(self, method: str, path: str, status_code: int, duration: float, **kwargs):
        """Log API request with metrics."""
        context = self._get_context()
        context.update({
            "api_request": True,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration * 1000
        })
        filtered_context = SecurityFilter.filter_sensitive_data({**context, **kwargs})
        self.logger.info(f"{method} {path} - {status_code} ({duration:.2f}ms)", **filtered_context)
    
    def external_service_call(self, service: str, operation: str, duration: float, success: bool, **kwargs):
        """Log external service call."""
        context = self._get_context()
        context.update({
            "external_service": True,
            "service": service,
            "operation": operation,
            "duration_ms": duration * 1000,
            "success": success
        })
        filtered_context = SecurityFilter.filter_sensitive_data({**context, **kwargs})
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"External service {service}.{operation} - {status} ({duration:.2f}ms)", **filtered_context)
    
    def database_query(self, query_type: str, table: str, duration: float, **kwargs):
        """Log database query performance."""
        context = self._get_context()
        context.update({
            "database_query": True,
            "query_type": query_type,
            "table": table,
            "duration_ms": duration * 1000
        })
        filtered_context = SecurityFilter.filter_sensitive_data({**context, **kwargs})
        self.logger.info(f"Database {query_type} on {table} ({duration:.2f}ms)", **filtered_context)


class LoggerMiddleware:
    """Middleware for automatic request logging."""
    
    def __init__(self, app):
        self.app = app
        self.logger = get_logger("middleware")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = datetime.now()
            
            # Log request start
            self.logger.info(
                f"Request started: {scope['method']} {scope['path']}",
                method=scope['method'],
                path=scope['path'],
                client=scope.get('client', ['unknown', 0])[0]
            )
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    duration = (datetime.now() - start_time).total_seconds()
                    self.logger.api_request(
                        method=scope['method'],
                        path=scope['path'],
                        status_code=message["status"],
                        duration=duration,
                        client=scope.get('client', ['unknown', 0])[0]
                    )
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


# Global logger cache
_loggers = {}


def get_logger(name: str) -> StructuredLogger:
    """Get or create a logger instance."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def setup_logging():
    """Setup global logging configuration."""
    # Disable some noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    
    # Setup root logger
    root_logger = get_logger("insight_miner")
    root_logger.info("Logging system initialized", environment=settings.application.environment)


# Initialize logging on import
setup_logging()