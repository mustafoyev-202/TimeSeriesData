# Running Kafka on Windows

This guide explains how to set up and run Apache Kafka on Windows using three simple commands.

## Prerequisites
- **Java**: Kafka requires Java 8 or later. Ensure Java is installed by running:
  ```powershell
  java -version
  ```
- **Kafka Downloaded**: Extract Kafka to `C:\kafka` (Recommended for avoiding long path issues).

## Steps to Run Kafka

### 1️⃣ Start Zookeeper
Kafka requires Zookeeper to manage brokers. Run the following command in **PowerShell**:
```powershell
cd C:\kafka
.\bin\windows\zookeeper-server-start.bat config\zookeeper.properties
```

### 2️⃣ Start Kafka Broker
Open a **new PowerShell window** and run:
```powershell
cd C:\kafka
.\bin\windows\kafka-server-start.bat config\server.properties
```

### 3️⃣ Run Your Kafka Application
Once Kafka is running, start your Python application (or producer/consumer):
```powershell
cd C:\Users\user\PycharmProjects\mohirdev_1
python kafka_streaming/run.py
```

## Verifying Kafka is Running
After starting Kafka, check if it’s listening on port **9092**:
```powershell
netstat -ano | findstr :9092
```
If you see an output like:
```
TCP    127.0.0.1:9092     LISTENING     <PID>
```
Kafka is running successfully!

## Optional: Managing Kafka Topics
### List all topics:
```powershell
cd C:\kafka
.\bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092
```
### Create a new topic (e.g., `anomalies`):
```powershell
.\bin\windows\kafka-topics.bat --create --topic anomalies --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

## Troubleshooting
- **Error: "The input line is too long"** → Move Kafka to `C:\kafka` and retry.
- **Kafka not starting?** → Ensure Zookeeper is running first.
- **Python Kafka producer fails?** → Check `bootstrap_servers=["localhost:9092"]` in your Python code.

---
Now you're ready to run Kafka smoothly on Windows! 🚀

