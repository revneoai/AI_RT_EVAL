from kafka import KafkaConsumer
import json

class MessageConsumer:
    """Kafka message consumer"""
    def __init__(self, bootstrap_servers: str, topic: str):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

    async def consume_messages(self):
        """Consume messages from Kafka topic"""
        try:
            for message in self.consumer:
                yield message.value
        except Exception as e:
            # Handle error or retry
            raise e
