import json
import os
import logging
from joblib import load
from multiprocessing import Process
import numpy as np
from utils import create_producer, create_consumer


def detect_anomalies(
    kafka_broker,
    transactions_topic,
    anomalies_topic,
    consumer_group,
    model_path,
    process_id=0,
):
    """Detect anomalies in transaction data using an isolation forest model."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - Detector#{process_id} - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(f"anomaly_detector_{process_id}")

    # Load the model
    try:
        logger.info(f"Loading model from {model_path}")
        clf = load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        return

    # Create Kafka consumer and producer
    consumer = create_consumer(
        kafka_broker=kafka_broker, topic=transactions_topic, group_id=consumer_group
    )

    producer = create_producer(kafka_broker)

    if consumer is None or producer is None:
        logger.error("Failed to initialize Kafka client(s), exiting")
        return

    logger.info(f"Starting anomaly detection on topic: {transactions_topic}")
    processed_count = 0
    anomaly_count = 0

    try:
        while True:
            message = consumer.poll(timeout=1.0)
            if message is None:
                continue
            if message.error():
                logger.error(f"Consumer error: {message.error()}")
                continue

            # Process the message
            try:
                # Parse the message
                record = json.loads(message.value().decode("utf-8"))
                data = record["data"]

                # Make prediction
                prediction = clf.predict(data)
                processed_count += 1
    
                # Log progress occasionally
                if processed_count % 100 == 0:
                    logger.info(
                        f"Processed {processed_count} messages, found {anomaly_count} anomalies"
                    )

                # If an anomaly is detected, send it to anomalies topic
                if prediction[0] == -1:
                    anomaly_count += 1
                    # Get the anomaly score
                    score = clf.score_samples(data)
                    record["score"] = np.round(score, 3).tolist()
                    record["detector_id"] = process_id

                    # Convert to JSON and send to anomalies topic
                    record_json = json.dumps(record).encode("utf-8")

                    producer.produce(topic=anomalies_topic, value=record_json)
                    producer.flush()
                    logger.info(
                        f"Anomaly detected! ID: {record['id']}, Score: {score[0]}"
                    )

                # Commit offset after processing
                consumer.commit()

            except json.JSONDecodeError:
                logger.error("Failed to parse message as JSON")
            except Exception as e:
                logger.exception(f"Error processing message: {e}")

    except KeyboardInterrupt:
        logger.info("Shutting down anomaly detector")
    finally:
        consumer.close()
        producer.flush()
        logger.info(
            f"Anomaly detector stopped after processing {processed_count} messages"
        )
