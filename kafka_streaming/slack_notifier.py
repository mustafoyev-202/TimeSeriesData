import json
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from utils import create_consumer


def run_slack_notifier(
    kafka_broker, anomalies_topic, anomalies_consumer_group, slack_token, slack_channel
):
    """
    Consume anomaly messages from Kafka and send them to Slack.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("slack_notifier")

    # Initialize Slack client
    client = WebClient(token=slack_token)

    # Test Slack authentication
    try:
        client.auth_test()
        logger.info(
            f"Successfully connected to Slack, posting to channel: {slack_channel}"
        )
    except SlackApiError as e:
        logger.error(
            f"Slack connection failed: {e.response.get('error', 'Unknown error')}"
        )
        return

    # Initialize Kafka consumer
    consumer = create_consumer(
        kafka_broker=kafka_broker,
        topic=anomalies_topic,
        group_id=anomalies_consumer_group,
    )

    if consumer is None:
        logger.error("Failed to create Kafka consumer, exiting")
        return

    logger.info(f"Starting to consume from topic: {anomalies_topic}")
    try:
        while True:
            message = consumer.poll(timeout=1.0)
            if message is None:
                continue
            if message.error():
                logger.error(f"Consumer error: {message.error()}")
                continue

            # Parse the message
            try:
                record = json.loads(message.value().decode("utf-8"))

                # Construct the Slack message
                slack_message = (
                    f"*ANOMALY DETECTED*"
                    f"ID: {record.get('id', 'Unknown')}"
                    f"Score: {record.get('score', 'N/A')}"
                    f"Data: {record.get('data', 'No data')}"
                    f"Time: {record.get('timestamp', record.get('current_time', 'N/A'))}"
                )

                # Send the message to Slack
                response = client.chat_postMessage(
                    channel=slack_channel, text=slack_message, unfurl_links=False
                )
                logger.info(
                    f"Sent anomaly notification to Slack, message ts: {response['ts']}"
                )

                # Commit the offset after successful processing
                consumer.commit()
            except json.JSONDecodeError:
                logger.error("Failed to parse message as JSON")
            except SlackApiError as e:
                logger.error(
                    f"Failed to send to Slack: {e.response.get('error', 'Unknown error')}"
                )
            except Exception as e:
                logger.exception(f"Unexpected error processing message: {e}")

    except KeyboardInterrupt:
        logger.info("Shutting down Slack notifier")
    finally:
        consumer.close()
        logger.info("Slack notifier stopped")
