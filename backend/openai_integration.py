import os
import openai
import redis.asyncio as redis
import json
from loguru import logger

class OpenAIIntegration:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self.default_model = "gpt-3.5-turbo"
        self.fallback_model = "gpt-3.5-turbo-instruct" # Example fallback for completion

    async def get_chat_completion(self, prompt: str, stream: bool = False, model: str = None):
        model = model or self.default_model
        cache_key = f"openai_chat:{prompt}:{model}"
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for OpenAI chat completion: {prompt}")
            return json.loads(cached_result)

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=stream
            )
            if stream:
                return self._handle_stream_response(response, cache_key)
            else:
                result = response.choices[0].message.content
                await self.redis_client.set(cache_key, json.dumps(result), ex=3600) # Cache for 1 hour
                return result
        except openai.error.RateLimitError:
            logger.warning(f"Rate limit hit for model {model}. Falling back to {self.fallback_model}")
            if model == self.fallback_model:
                raise # Avoid infinite loop if fallback also hits rate limit
            return await self.get_chat_completion(prompt, stream, self.fallback_model)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _handle_stream_response(self, response, cache_key):
        full_response_content = ""
        for chunk in response:
            content = chunk["choices"][0].delta.get("content", "")
            full_response_content += content
            yield content
        await self.redis_client.set(cache_key, json.dumps(full_response_content), ex=3600)

openai_integration = OpenAIIntegration()
