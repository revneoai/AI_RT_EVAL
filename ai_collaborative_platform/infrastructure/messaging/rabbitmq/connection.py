import aio_pika
from typing import Optional

class RabbitMQConnection:
    """RabbitMQ connection manager"""
    def __init__(self, url: str):
        self.url = url
        self.connection: Optional[aio_pika.Connection] = None
        
    async def connect(self):
        """Establish connection to RabbitMQ"""
        self.connection = await aio_pika.connect_robust(self.url)
        
    async def close(self):
        """Close RabbitMQ connection"""
        if self.connection:
            await self.connection.close()
