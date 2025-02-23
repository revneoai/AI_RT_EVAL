import redis.asyncio as redis
from typing import Optional, Any

class RedisAdapter:
    """Redis cache adapter"""
    def __init__(self, redis_url: str):
        self.client = redis.from_url(redis_url)

    async def set(self, key: str, value: Any, expire: int = 3600):
        await self.client.set(key, value, ex=expire)

    async def get(self, key: str) -> Optional[Any]:
        return await self.client.get(key)
