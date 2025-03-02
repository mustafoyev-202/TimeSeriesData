import os
from dotenv import load_dotenv

load_dotenv()

KAFKA_BROKER = "localhost:9092"
TRANSACTIONS_TOPIC = "transactions"
ANOMALIES_TOPIC = "anomalies"
TRANSACTIONS_CONSUMER_GROUP = "anomaly-detectors"
ANOMALIES_CONSUMER_GROUP = "slack-notifier"
NUM_PARTITIONS = 4  # Should match the number of partitions in your Kafka topic

# Data generation settings
DELAY = 0.1  # Seconds between generated data points
OUTLIERS_GENERATION_PROBABILITY = 0.05

# Slack settings
SLACK_API_TOKEN = os.getenv("SLACK_API_TOKEN")
SLACK_CHANNEL = "#all-mohirdev"

# Model path
MODEL_PATH = os.path.abspath("model\isolation_forest_model.pkl")
