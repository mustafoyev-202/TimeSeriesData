# 🔥 Ambient Temperature Anomaly Detection 🔥

## 🚀 Overview
This project implements **cutting-edge anomaly detection** on ambient temperature data using a **hybrid approach**: an ensemble of **Isolation Forest** (statistical method) and an **LSTM Autoencoder** (deep learning model). The goal is to identify **unusual temperature fluctuations** that could indicate system failures.

## 🛠️ Technologies Used
- **Python** 🐍
- **Pandas, NumPy** for data handling 📊
- **Matplotlib, Seaborn** for visualization 🎨
- **Scikit-learn** for Isolation Forest 🏗️
- **TensorFlow/Keras** for LSTM Autoencoder 🧠

## 📂 Dataset
We use the **Numenta Anomaly Benchmark (NAB)** dataset:
- **File:** `ambient_temperature_system_failure.csv`
- **Source:** [NAB GitHub Repo](https://github.com/numenta/NAB)
- **Data:** Timestamped temperature readings with known anomalies.

## 📊 Workflow
### 🔍 1. Data Preprocessing
- Load and clean dataset ✅
- Handle missing values and duplicates ✅
- Extract **rolling statistics, time-based features, and lagged values** ✅
- Normalize data using **MinMaxScaler** ✅

### 🧠 2. Model Training
#### 📡 Isolation Forest (Anomaly Detection)
- Trained with **contamination = 1%** (expected anomaly ratio)
- Uses decision function scores to detect anomalies

#### 🤖 LSTM Autoencoder
- Converts sequential data into a **compressed representation**
- Reconstructs input and detects anomalies based on **reconstruction error**
- **Threshold:** 95th percentile of reconstruction MSE

### 🎯 3. Anomaly Detection & Evaluation
- **Both models generate anomaly scores** 🚨
- **Ensemble approach:** Combines results from both methods for higher accuracy ✅
- **Visualizations:**
  - Original data with anomalies marked (3 subplots: Isolation Forest, LSTM, Ensemble) 📈
- **Results saved:** `anomaly_detection_results.csv`

## 📸 Key Visualizations
✅ **Raw Data Plot** (Before Processing)  
✅ **Feature Correlation Heatmap**  
✅ **Isolation Forest Anomalies**  
✅ **LSTM Autoencoder Training History**  
✅ **Final Anomaly Detection Plots**  

## 🚀 Results
🎯 **Anomalies detected by Isolation Forest:** `{results['iforest_anomaly'].sum()}`  
🎯 **Anomalies detected by LSTM Autoencoder:** `{results['ae_anomaly'].dropna().astype(int).sum()}`  
🎯 **Anomalies detected by Ensemble:** `{results['ensemble_anomaly'].sum()}`  

📌 **Ensemble method improves detection accuracy by leveraging the strengths of both models.**

## 🏁 How to Run
### 1️⃣ Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### 2️⃣ Run the Script
```bash
python anomaly_detection.py
```

### 3️⃣ View Results
- **CSV Output:** `anomaly_detection_results.csv`
- **Plots:** `anomaly_detection_results.png`, `feature_correlation.png`, etc.

## 🏆 Conclusion
This project provides a **powerful, hybrid anomaly detection system** that can be applied to industrial monitoring, IoT sensor networks, and predictive maintenance. 🚀🔥

---
🔗 **Developed by:** Baxtiyor Mustafoyev 

