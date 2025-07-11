import pytest
from unittest.mock import AsyncMock, patch
from ..sentiment_analysis import SentimentAnalyzer

@pytest.fixture
def sentiment_analyzer_instance():
    return SentimentAnalyzer()

@pytest.mark.asyncio
async def test_analyze_with_google_success(sentiment_analyzer_instance):
    with patch('google.cloud.language_v1.LanguageServiceClient') as MockGoogleClient:
        mock_sentiment = MockGoogleClient.return_value.analyze_sentiment.return_value
        mock_sentiment.document_sentiment.score = 0.8
        mock_sentiment.document_sentiment.magnitude = 0.9
        
        result = await sentiment_analyzer_instance._analyze_with_google("This is a great text.")
        assert result["score"] == 0.8
        assert result["magnitude"] == 0.9
        assert result["provider"] == "google"

@pytest.mark.asyncio
async def test_analyze_with_aws_success(sentiment_analyzer_instance):
    with patch('boto3.client') as MockBoto3Client:
        mock_comprehend = MockBoto3Client.return_value
        mock_comprehend.detect_sentiment.return_value = {
            "SentimentScore": {"Positive": 0.9, "Negative": 0.1, "Neutral": 0.0, "Mixed": 0.0}
        }
        
        result = await sentiment_analyzer_instance._analyze_with_aws("This is a great text.")
        assert result["score"] == 0.8 # 0.9 - 0.1
        assert result["magnitude"] == 1.0 # 0.9 + 0.1
        assert result["provider"] == "aws"

@pytest.mark.asyncio
async def test_sentiment_analysis_cache_hit(sentiment_analyzer_instance):
    with patch('redis.asyncio.Redis.get', new_callable=AsyncMock) as mock_redis_get:
        mock_redis_get.return_value = "{'score': 0.7, 'magnitude': 0.8, 'provider': 'cache'}"
        
        result = await sentiment_analyzer_instance.analyze_sentiment("Cached text.")
        assert result["score"] == 0.7
        assert result["provider"] == "cache"
        mock_redis_get.assert_called_once_with("sentiment:Cached text.")

@pytest.mark.asyncio
async def test_sentiment_analysis_fallback(sentiment_analyzer_instance):
    with patch('google.cloud.language_v1.LanguageServiceClient') as MockGoogleClient,
         patch('boto3.client') as MockBoto3Client,
         patch('redis.asyncio.Redis.get', new_callable=AsyncMock) as mock_redis_get,
         patch('redis.asyncio.Redis.set', new_callable=AsyncMock) as mock_redis_set:
        
        mock_redis_get.return_value = None # No cache hit
        MockGoogleClient.return_value.analyze_sentiment.side_effect = Exception("Google API error")
        mock_comprehend = MockBoto3Client.return_value
        mock_comprehend.detect_sentiment.return_value = {
            "SentimentScore": {"Positive": 0.9, "Negative": 0.1, "Neutral": 0.0, "Mixed": 0.0}
        }

        result = await sentiment_analyzer_instance.analyze_sentiment("Text for fallback.")
        assert result["provider"] == "aws"
        mock_redis_set.assert_called_once()
