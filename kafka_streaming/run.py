
if __name__ == "__main__":
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from multiprocessing import Process
    import time

    # Import components
    from data_generator import run_data_generator
    from anomaly_detector import detect_anomalies
    from main import launch_anomaly_detectors
    from slack_notifier import run_slack_notifier

    # Import settings
    from settings import (
        KAFKA_BROKER,
        TRANSACTIONS_TOPIC,
        ANOMALIES_TOPIC,
        TRANSACTIONS_CONSUMER_GROUP,
        ANOMALIES_CONSUMER_GROUP,
        NUM_PARTITIONS,
        DELAY,
        OUTLIERS_GENERATION_PROBABILITY,
        SLACK_API_TOKEN,
        SLACK_CHANNEL,
        MODEL_PATH,
    )

    # Start data generator
    data_gen_process = Process(
        target=run_data_generator,
        args=(KAFKA_BROKER, TRANSACTIONS_TOPIC, DELAY, OUTLIERS_GENERATION_PROBABILITY),
    )
    data_gen_process.daemon = True
    data_gen_process.start()
    logging.info("Started data generator process")

    # Start anomaly detectors (one per partition)
    detector_processes = launch_anomaly_detectors(
        KAFKA_BROKER,
        TRANSACTIONS_TOPIC,
        ANOMALIES_TOPIC,
        TRANSACTIONS_CONSUMER_GROUP,
        MODEL_PATH,
        NUM_PARTITIONS,
    )

    # Start Slack notifier
    slack_process = Process(
        target=run_slack_notifier,
        args=(
            KAFKA_BROKER,
            ANOMALIES_TOPIC,
            ANOMALIES_CONSUMER_GROUP,
            SLACK_API_TOKEN,
            SLACK_CHANNEL,
        ),
    )
    slack_process.daemon = True
    slack_process.start()
    logging.info("Started Slack notifier process")

    # Keep the main process running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down all processes")
