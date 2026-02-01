"""
Alpha Platform - Kafka Connection

Handles Kafka/Redpanda producer and consumer.
"""

import os
import json
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError


@dataclass
class KafkaConfig:
    """Kafka configuration."""
    bootstrap_servers: str = "localhost:19092"
    group_id: str = "alpha-platform"

    @classmethod
    def from_env(cls) -> "KafkaConfig":
        """Load configuration from environment variables."""
        return cls(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"),
            group_id=os.getenv("KAFKA_GROUP_ID", "alpha-platform"),
        )


class AlphaProducer:
    """Kafka producer for sending messages."""

    def __init__(self, config: Optional[KafkaConfig] = None):
        if config is None:
            config = KafkaConfig.from_env()

        self.config = config
        self.producer = KafkaProducer(
            bootstrap_servers=config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
        )

    def send(self, topic: str, value: Dict[str, Any], key: Optional[str] = None):
        """
        Send a message to a topic.

        Args:
            topic: Kafka topic name
            value: Message payload (dict, will be JSON serialized)
            key: Optional message key
        """
        future = self.producer.send(topic, value=value, key=key)
        return future

    def send_sync(self, topic: str, value: Dict[str, Any], key: Optional[str] = None, timeout: float = 10.0):
        """Send a message and wait for confirmation."""
        future = self.send(topic, value, key)
        return future.get(timeout=timeout)

    def flush(self):
        """Flush all pending messages."""
        self.producer.flush()

    def close(self):
        """Close the producer."""
        self.producer.close()


class AlphaConsumer:
    """Kafka consumer for receiving messages."""

    def __init__(self, topics: List[str], config: Optional[KafkaConfig] = None, group_id: Optional[str] = None):
        if config is None:
            config = KafkaConfig.from_env()

        self.config = config
        self.topics = topics

        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=config.bootstrap_servers,
            group_id=group_id or config.group_id,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
        )

    def poll(self, timeout_ms: int = 1000) -> List[Dict[str, Any]]:
        """
        Poll for new messages.

        Returns list of messages with topic, key, value.
        """
        records = self.consumer.poll(timeout_ms=timeout_ms)
        messages = []

        for topic_partition, record_list in records.items():
            for record in record_list:
                messages.append({
                    "topic": record.topic,
                    "key": record.key.decode('utf-8') if record.key else None,
                    "value": record.value,
                    "timestamp": record.timestamp,
                })

        return messages

    def listen(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Listen for messages continuously.

        Args:
            callback: Function called for each message (topic, value)
        """
        for message in self.consumer:
            callback(message.topic, message.value)

    def close(self):
        """Close the consumer."""
        self.consumer.close()


def test_kafka_connection(config: Optional[KafkaConfig] = None) -> bool:
    """Test Kafka connection by sending and receiving a test message."""
    if config is None:
        config = KafkaConfig.from_env()

    test_topic = "test.connection"
    test_message = {"test": "hello", "source": "alpha-platform"}

    # Send test message
    producer = AlphaProducer(config)
    try:
        producer.send_sync(test_topic, test_message, timeout=5.0)
        print(f"Sent test message to '{test_topic}'")
        producer.close()
    except KafkaError as e:
        print(f"Failed to send: {e}")
        producer.close()
        return False

    # Receive test message
    consumer = AlphaConsumer([test_topic], config, group_id="test-connection-group")
    try:
        messages = consumer.poll(timeout_ms=5000)
        if messages:
            print(f"Received {len(messages)} message(s)")
            for msg in messages:
                print(f"  Topic: {msg['topic']}, Value: {msg['value']}")
        else:
            print("No messages received (this is OK for first run)")
        consumer.close()
    except KafkaError as e:
        print(f"Failed to receive: {e}")
        consumer.close()
        return False

    return True