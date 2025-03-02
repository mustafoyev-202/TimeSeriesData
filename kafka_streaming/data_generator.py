import json
import random
import socket
import time
from datetime import datetime
import logging
import numpy as np
from utils import create_producer


def run_data_generator(
    kafka_broker, transactions_topic, delay=0.1, outlier_probability=0.05
):
    """
    Generate and send transaction data to a Kafka topic.

    Parameters:
        kafka_broker (str): Kafka broker address.
        transactions_topic (str): Kafka topic for publishing data.
        delay (float): Time interval (seconds) between messages.
        outlier_probability (float): Probability of generating an outlier.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("data_generator")

    # Initialize Kafka producer
    producer = create_producer(kafka_broker)
    if producer is None:
        logger.error("Failed to initialize producer, exiting data generator")
        return

    logger.info("Starting data generation loop")
    _id = 0

    try:
        while True:
            # Generate synthetic transaction data
            if random.random() <= outlier_probability:
                data_point = np.random.uniform(
                    low=-4, high=4, size=(1, 14)
                )  # Outlier data
                is_outlier = True
            else:
                data_point = np.random.randn(1, 14)  # Normal data
                is_outlier = False

            data_point = np.round(data_point, 3).tolist()
            current_time = datetime.utcnow().isoformat()

            # Create record
            record = {
                "id": _id,
                "data": data_point,
                "timestamp": current_time,
                "generator_info": {
                    "expected_outlier": is_outlier,
                    "hostname": socket.gethostname(),
                },
            }
            record_json = json.dumps(record).encode("utf-8")

            # Define delivery callback function
            def delivery_callback(err, msg):
                if err:
                    logger.error(f"Message delivery failed: {err}")
                else:
                    logger.debug(
                        f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}"
                    )

            # Send message to Kafka
            producer.produce(
                topic=transactions_topic, value=record_json, callback=delivery_callback
            )

            # Flush every 10 messages
            if _id % 10 == 0:
                producer.flush()

            _id += 1
            time.sleep(delay)

    except KeyboardInterrupt:
        logger.info("Shutting down data generator")
    finally:
        producer.flush()
        logger.info("Data generator stopped")
