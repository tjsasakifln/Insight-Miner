import os
import asyncio
import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import redis.asyncio as redis
from google.cloud import language_v1
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from circuitbreaker import circuit
import aiohttp
import time

from .config import settings
from .logger import get_logger

logger = get_logger(__name__)

class SentimentAnalyzer:
    """Enterprise-grade sentiment analysis with circuit breakers and monitoring."""
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.redis.connection_string)
        self.cache_ttl = 3600  # 1 hour cache
        self.max_text_length = 5000
        self.request_timeout = 30
        
        # Initialize providers
        self._init_google_client()
        self._init_aws_client()
        
        # Circuit breaker configuration
        self.circuit_breaker_config = {
            'failure_threshold': 5,
            'recovery_timeout': 60,
            'expected_exception': Exception
        }
        
        # Metrics
        self.metrics = {
            'requests_total': 0,
            'cache_hits': 0,
            'google_requests': 0,
            'aws_requests': 0,
            'failures': 0
        }
    
    def _init_google_client(self):
        """Initialize Google Cloud Language client with proper error handling."""
        try:
            if settings.external_services.google_credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = settings.external_services.google_credentials_path
            
            self.google_client = language_v1.LanguageServiceClient()
            logger.info("Google Cloud Language client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Google client: {e}")
            self.google_client = None
    
    def _init_aws_client(self):
        """Initialize AWS Comprehend client with proper error handling."""
        try:
            self.aws_client = boto3.client(
                "comprehend",
                aws_access_key_id=settings.external_services.aws_access_key_id,
                aws_secret_access_key=settings.external_services.aws_secret_access_key,
                region_name=settings.external_services.aws_region
            )
            logger.info("AWS Comprehend client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AWS client: {e}")
            self.aws_client = None
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate secure cache key for text."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"sentiment:v2:{text_hash}"
    
    def _validate_text(self, text: str) -> str:
        """Validate and sanitize input text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text = text.strip()
        
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
            logger.warning(f"Text truncated to {self.max_text_length} characters")
        
        return text
    
    @circuit(failure_threshold=5, recovery_timeout=60, expected_exception=Exception)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ClientError, Exception))
    )
    async def _analyze_with_google(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Google Cloud Natural Language API."""
        if not self.google_client:
            raise Exception("Google client not initialized")
        
        try:
            start_time = time.time()
            
            document = language_v1.Document(
                content=text, 
                type_=language_v1.Document.Type.PLAIN_TEXT
            )
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.google_client.analyze_sentiment,
                {'document': document}
            )
            
            sentiment = response.document_sentiment
            
            result = {
                "score": float(sentiment.score),
                "magnitude": float(sentiment.magnitude),
                "provider": "google",
                "confidence": abs(sentiment.score),
                "response_time": time.time() - start_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.metrics['google_requests'] += 1
            logger.info(f"Google analysis completed in {result['response_time']:.2f}s")
            
            return result
            
        except Exception as e:
            self.metrics['failures'] += 1
            logger.error(f"Google Cloud NL API error: {e}")
            raise
    
    @circuit(failure_threshold=5, recovery_timeout=60, expected_exception=Exception)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ClientError, Exception))
    )
    async def _analyze_with_aws(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using AWS Comprehend."""
        if not self.aws_client:
            raise Exception("AWS client not initialized")
        
        try:
            start_time = time.time()
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.aws_client.detect_sentiment,
                text,
                "en"
            )
            
            sentiment_scores = response["SentimentScore"]
            
            # Convert AWS sentiment to normalized score (-1 to 1)
            score = sentiment_scores["Positive"] - sentiment_scores["Negative"]
            magnitude = sentiment_scores["Positive"] + sentiment_scores["Negative"]
            
            result = {
                "score": float(score),
                "magnitude": float(magnitude),
                "provider": "aws",
                "confidence": max(sentiment_scores.values()),
                "response_time": time.time() - start_time,
                "timestamp": datetime.utcnow().isoformat(),
                "raw_sentiment": response["Sentiment"]
            }
            
            self.metrics['aws_requests'] += 1
            logger.info(f"AWS analysis completed in {result['response_time']:.2f}s")
            
            return result
            
        except ClientError as e:
            self.metrics['failures'] += 1
            logger.error(f"AWS Comprehend API error: {e}")
            raise
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get sentiment analysis from cache."""
        try:
            cached_result = await self.redis_client.get(cache_key)
            if cached_result:
                self.metrics['cache_hits'] += 1
                logger.debug("Cache hit for sentiment analysis")
                return json.loads(cached_result)  # FIXED: Using json.loads instead of eval
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    async def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Save sentiment analysis to cache."""
        try:
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(result)
            )
            logger.debug("Result cached successfully")
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment with multiple providers, caching, and circuit breakers.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
            
        Raises:
            ValueError: If text is invalid
            HTTPException: If all providers fail
        """
        try:
            # Validate input
            validated_text = self._validate_text(text)
            
            # Update metrics
            self.metrics['requests_total'] += 1
            
            # Check cache first
            cache_key = self._generate_cache_key(validated_text)
            cached_result = await self._get_from_cache(cache_key)
            
            if cached_result:
                return cached_result
            
            # Try providers in order
            providers = []
            if self.google_client:
                providers.append(("google", self._analyze_with_google))
            if self.aws_client:
                providers.append(("aws", self._analyze_with_aws))
            
            if not providers:
                raise HTTPException(
                    status_code=503,
                    detail="No sentiment analysis providers available"
                )
            
            last_exception = None
            
            for provider_name, provider_func in providers:
                try:
                    logger.info(f"Attempting sentiment analysis with {provider_name}")
                    result = await provider_func(validated_text)
                    
                    # Cache successful result
                    await self._save_to_cache(cache_key, result)
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Provider {provider_name} failed: {e}")
                    continue
            
            # All providers failed
            self.metrics['failures'] += 1
            logger.error("All sentiment analysis providers failed")
            
            raise HTTPException(
                status_code=503,
                detail=f"All sentiment analysis providers failed. Last error: {last_exception}"
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in sentiment analysis: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal error in sentiment analysis"
            )
    
    async def analyze_batch(self, texts: list[str]) -> list[Dict[str, Any]]:
        """Analyze sentiment for multiple texts concurrently."""
        if not texts:
            return []
        
        if len(texts) > 100:
            raise HTTPException(
                status_code=400,
                detail="Batch size cannot exceed 100 texts"
            )
        
        # Process texts concurrently
        tasks = [self.analyze_sentiment(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch analysis failed for text {i}: {result}")
                processed_results.append({
                    "error": str(result),
                    "text_index": i,
                    "provider": "none"
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            **self.metrics,
            "cache_hit_ratio": (
                self.metrics['cache_hits'] / max(self.metrics['requests_total'], 1)
            ),
            "google_available": self.google_client is not None,
            "aws_available": self.aws_client is not None,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on sentiment analysis service."""
        health_status = {
            "service": "sentiment_analysis",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "providers": {}
        }
        
        # Test Google provider
        if self.google_client:
            try:
                await self._analyze_with_google("test")
                health_status["providers"]["google"] = "healthy"
            except Exception as e:
                health_status["providers"]["google"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
        
        # Test AWS provider
        if self.aws_client:
            try:
                await self._analyze_with_aws("test")
                health_status["providers"]["aws"] = "healthy"
            except Exception as e:
                health_status["providers"]["aws"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
        
        return health_status


# Global instance
sentiment_analyzer = SentimentAnalyzer()
