import os
import asyncio
import redis.asyncio as redis
from google.cloud import language_v1
import boto3
from botocore.exceptions import ClientError
from loguru import logger

class SentimentAnalyzer:
    def __init__(self):
        self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self.google_client = language_v1.LanguageServiceClient()
        self.aws_client = boto3.client(
            "comprehend",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION_NAME", "us-east-1")
        )

    async def _analyze_with_google(self, text: str):
        try:
            document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
            sentiment = self.google_client.analyze_sentiment(request={'document': document}).document_sentiment
            return {"score": sentiment.score, "magnitude": sentiment.magnitude, "provider": "google"}
        except Exception as e:
            logger.error(f"Google Cloud NL API error: {e}")
            raise

    async def _analyze_with_aws(self, text: str):
        try:
            response = self.aws_client.detect_sentiment(Text=text, LanguageCode="en") # Assuming English for now
            sentiment = response["SentimentScore"]
            return {"score": sentiment["Positive"] - sentiment["Negative"], "magnitude": sentiment["Positive"] + sentiment["Negative"], "provider": "aws"}
        except ClientError as e:
            logger.error(f"AWS Comprehend API error: {e}")
            raise

    async def analyze_sentiment(self, text: str):
        cache_key = f"sentiment:{text}"
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for sentiment analysis: {text}")
            return eval(cached_result) # Using eval for simplicity, consider json.loads for production

        providers = [self._analyze_with_google, self._analyze_with_aws]
        for i, provider in enumerate(providers):
            for attempt in range(3): # Retry up to 3 times
                try:
                    result = await provider(text)
                    await self.redis_client.set(cache_key, str(result), ex=3600) # Cache for 1 hour
                    return result
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed with {provider.__name__}: {e}")
                    if attempt < 2: # Don't wait on last attempt
                        await asyncio.sleep(2 ** attempt) # Exponential backoff
            if i < len(providers) - 1: # Fallback to next provider if current one fails all retries
                logger.error(f"All retries failed for {provider.__name__}. Falling back to next provider.")
        
        raise HTTPException(status_code=500, detail="All sentiment analysis providers failed.")

sentiment_analyzer = SentimentAnalyzer()
