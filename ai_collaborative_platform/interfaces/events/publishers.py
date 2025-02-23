from typing import Dict, Any
from abc import ABC, abstractmethod

class EventPublisher(ABC):
    """Abstract event publisher"""
    @abstractmethod
    async def publish(self, topic: str, event: Dict[str, Any]):
        """Publish event to topic"""
        pass

class KafkaEventPublisher(EventPublisher):
    """Kafka implementation of event publisher"""
    def __init__(self, producer):
        self.producer = producer
        
    async def publish(self, topic: str, event: Dict[str, Any]):
        await self.producer.send_message(topic, event)
