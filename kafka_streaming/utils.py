import logging
import socket
from confluent_kafka import Producer, Consumer, KafkaException


def create_producer(kafka_broker):
    """
    Create and return a Kafka producer with optimized settings.
    """
    try:
        producer = Producer(
            {
                "bootstrap.servers": kafka_broker,  # Kafka broker address
                "client.id": socket.gethostname(),  # Unique client identifier
                "enable.idempotence": True,  # Ensures exactly-once semantics
                "compression.type": "lz4",  # Efficient compression
                "batch.size": 65536,  # Batching messages for better throughput
                "linger.ms": 5,  # Delay in milliseconds to allow batching
                "acks": "all",  # Ensures all replicas acknowledge messages
                "retries": 5,  # Retry count on failure
                "delivery.timeout.ms": 3000,  # Max time for message delivery
                "statistics.interval.ms": 60000,  # Kafka statistics interval
            }
        )
        logging.info("Kafka producer created successfully")
        return producer
    except KafkaException as e:
        logging.exception("Failed to create Kafka producer: %s", str(e))
        return None


def create_consumer(
    kafka_broker, topic, group_id, auto_offset_reset="latest", enable_auto_commit=False
):
    """
    Create and return a Kafka consumer with specified settings.
    """
    try:
        consumer = Consumer(
            {
                "bootstrap.servers": kafka_broker,  # Kafka broker address
                "group.id": group_id,  # Consumer group ID
                "client.id": socket.gethostname(),  # Unique client identifier
                "isolation.level": "read_committed",  # Read only committed messages
                "auto.offset.reset": auto_offset_reset,  # Start position (latest or earliest)
                "enable.auto.commit": enable_auto_commit,  # Manual or automatic offset commit
                "max.poll.interval.ms": 300000,  # Max processing time per poll
                "session.timeout.ms": 30000,  # Timeout for consumer failure detection
                "heartbeat.interval.ms": 10000,  # Interval between heartbeats
            }
        )
        consumer.subscribe([topic])  # Subscribe to the given topic
        logging.info(f"Kafka consumer created and subscribed to topic: {topic}")
        return consumer
    except KafkaException as e:
        logging.exception("Failed to create Kafka consumer: %s", str(e))
        return None
