import redis
from typing import Any

class CacheManager:
    """Redis caching as specified in requirements"""
    def __init__(self):
        self.redis_client = redis.Redis()
