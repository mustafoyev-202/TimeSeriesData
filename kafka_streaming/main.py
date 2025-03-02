import os
import logging
from multiprocessing import Process
from anomaly_detector import detect_anomalies


def launch_anomaly_detectors(
    kafka_broker,
    transactions_topic,
    anomalies_topic,
    consumer_group,
    model_path,
    num_partitions,
):
    """Launch multiple anomaly detector processes."""
    processes = []

    for i in range(num_partitions):
        p = Process(
            target=detect_anomalies,
            args=(
                kafka_broker,
                transactions_topic,
                anomalies_topic,
                consumer_group,
                model_path,
                i,
            ),
        )
        p.daemon = True  # Ensure process terminates when main process does
        p.start()
        processes.append(p)
        logging.info(f"Started anomaly detector process #{i}")

    return processes
