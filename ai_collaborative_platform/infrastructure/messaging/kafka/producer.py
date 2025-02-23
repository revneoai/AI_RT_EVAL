from typing import Any, Dict
from kafka import KafkaProducer
import json

class MessageProducer:
    """Kafka message producer"""
    def __init__(self, bootstrap_servers: str):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    async def send_message(self, topic: str, message: Dict[str, Any]):
        """Send message to Kafka topic"""
        try:
            future = self.producer.send(topic, message)
            await future
        except Exception as e:
            # Handle error or retry
            raise e
