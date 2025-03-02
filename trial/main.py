import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ================ 1. DATA PREPROCESSING ================
print("1. DATA PREPROCESSING")

# Load data from NAB dataset
data = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ambient_temperature_system_failure.csv')
print(f"Dataset loaded with shape: {data.shape}")

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Exploratory Data Analysis
print("\nBasic statistics:")
print(data.describe())

# Check for missing values and duplicates
print(f"\nMissing values: {data.isnull().sum().sum()}")
print(f"Duplicate rows: {data.duplicated().sum()}")

# Remove duplicates and missing values if any
data_cleaned = data.dropna().drop_duplicates()
print(f"Clean dataset shape: {data_cleaned.shape}")

# Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(data_cleaned['timestamp'], data_cleaned['value'])
plt.title('Ambient Temperature System Data')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.grid(True)
plt.savefig('raw_data_visualization.png')
plt.close()

# ================ 2. FEATURE ENGINEERING ================
print("\n2. FEATURE ENGINEERING")

# Create rolling statistics
windows = [10, 20, 30]
for w in windows:
    data_cleaned[f'rolling_mean_{w}'] = data_cleaned['value'].rolling(window=w).mean()
    data_cleaned[f'rolling_std_{w}'] = data_cleaned['value'].rolling(window=w).std()

# Extract time-based features
data_cleaned['hour'] = data_cleaned['timestamp'].dt.hour
data_cleaned['day_of_week'] = data_cleaned['timestamp'].dt.dayofweek
data_cleaned['is_weekend'] = data_cleaned['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Create lag features
for lag in [1, 3, 5]:
    data_cleaned[f'lag_{lag}'] = data_cleaned['value'].shift(lag)

# Calculate difference features
data_cleaned['diff_1'] = data_cleaned['value'].diff()

# Drop rows with NaN values created by rolling windows and lags
data_processed = data_cleaned.dropna()
print(f"Processed dataset shape after feature engineering: {data_processed.shape}")

# Visualize feature correlations
plt.figure(figsize=(12, 10))
feature_cols = [col for col in data_processed.columns if col not in ['timestamp']]
sns.heatmap(data_processed[feature_cols].corr(), annot=False, cmap='viridis')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('feature_correlation.png')
plt.close()

# Normalize features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data_processed[feature_cols])
data_scaled = pd.DataFrame(scaled_features, columns=feature_cols)
data_scaled['timestamp'] = data_processed['timestamp'].values

# ================ 3. MODEL SELECTION & TRAINING ================
print("\n3. MODEL SELECTION & TRAINING")

# 3.1 Isolation Forest (Statistical approach)
print("Training Isolation Forest model...")
isolation_forest = IsolationForest(
    n_estimators=100, 
    contamination=0.01,  # Expected proportion of anomalies
    random_state=42,
    n_jobs=-1
)

# Extract features for isolation forest (exclude timestamp)
X_iforest = data_scaled.drop('timestamp', axis=1).values

# Train the model
isolation_forest.fit(X_iforest)

# Save the trained model
with open('isolation_forest_model.pkl', 'wb') as f:
    pickle.dump(isolation_forest, f)

print("Isolation Forest model trained and saved.")

# 3.2 LSTM Autoencoder (Deep Learning approach)
print("Preparing data for LSTM Autoencoder...")

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

# Prepare sequence data - use value column as the main signal
sequence_length = 20
X_train_scaled = data_scaled['value'].values.reshape(-1, 1)
X_train_seq = create_sequences(X_train_scaled, sequence_length)
print(f"Sequence data shape: {X_train_seq.shape}")

# Build LSTM Autoencoder
print("Building and training LSTM Autoencoder...")
input_dim = X_train_seq.shape[1:]  # (sequence_length, 1)

# Encoder
inputs = Input(shape=input_dim)
encoder = LSTM(32, activation='relu', return_sequences=True)(inputs)
encoder = LSTM(16, activation='relu', return_sequences=False)(encoder)
encoder = Dense(8, activation='relu')(encoder)

# Decoder
decoder = RepeatVector(sequence_length)(encoder)
decoder = LSTM(16, activation='relu', return_sequences=True)(decoder)
decoder = LSTM(32, activation='relu', return_sequences=True)(decoder)
decoder = TimeDistributed(Dense(1))(decoder)

# Autoencoder model
autoencoder = Model(inputs, decoder)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Train with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = autoencoder.fit(
    X_train_seq, X_train_seq,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# Save the trained autoencoder model
autoencoder.save('lstm_autoencoder_model.h5')
print("LSTM Autoencoder model trained and saved.")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Autoencoder Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('autoencoder_training_history.png')
plt.close()

# ================ 4. EVALUATION ================
print("\n4. EVALUATION")

# 4.1 Evaluate Isolation Forest
# Get anomaly predictions (-1 for anomalies, 1 for normal)
iforest_predictions = isolation_forest.predict(X_iforest)
# Convert to binary (1 for anomalies, 0 for normal)
iforest_anomalies = np.where(iforest_predictions == -1, 1, 0)
# Get anomaly scores
iforest_scores = -isolation_forest.decision_function(X_iforest)

# This is the fixed section of code for the error. Replace lines around 200-210 with this:

# 4.2 Evaluate LSTM Autoencoder
# Generate reconstructions
reconstructions = autoencoder.predict(X_train_seq)
# Compute reconstruction error (MSE)
mse = np.mean(np.square(X_train_seq - reconstructions), axis=(1, 2))
# Determine anomaly threshold (e.g., 95th percentile)
threshold = np.percentile(mse, 95)
print(f"Autoencoder anomaly threshold: {threshold}")

# Create a temporary series to hold autoencoder anomalies, initialized with zeros
ae_anomalies = np.zeros(len(data_processed))
# Mark anomalies where reconstruction error exceeds threshold
# Make sure indices match (sequence_length offset)
for i in range(len(mse)):
    if mse[i] > threshold:
        ae_anomalies[i + sequence_length] = 1

# Combine results into a DataFrame
results = pd.DataFrame({
    'timestamp': data_processed['timestamp'],
    'value': data_processed['value'],
    'iforest_anomaly': iforest_anomalies,
    'iforest_score': iforest_scores,
    'ae_anomaly': ae_anomalies  # Now correctly sized array
})

# Create ensemble model (combine both approaches)
results['ensemble_anomaly'] = np.where(
    (results['iforest_anomaly'] == 1) | (results['ae_anomaly'] == 1), 
    1, 0
)

# Count anomalies detected by each method
print(f"Anomalies detected by Isolation Forest: {results['iforest_anomaly'].sum()}")
print(f"Anomalies detected by LSTM Autoencoder: {results['ae_anomaly'].dropna().astype(int).sum()}")
print(f"Anomalies detected by Ensemble approach: {results['ensemble_anomaly'].sum()}")

# Visualize the results
plt.figure(figsize=(15, 10))

# Plot 1: Original data with Isolation Forest anomalies
plt.subplot(3, 1, 1)
plt.plot(results['timestamp'], results['value'], label='Temperature')
anomaly_points = results[results['iforest_anomaly'] == 1]
plt.scatter(anomaly_points['timestamp'], anomaly_points['value'], 
            color='red', label='Anomalies', s=50)
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)

# Plot 2: Original data with Autoencoder anomalies
plt.subplot(3, 1, 2)
plt.plot(results['timestamp'], results['value'], label='Temperature')
ae_anomaly_points = results[results['ae_anomaly'] == 1]
plt.scatter(ae_anomaly_points['timestamp'], ae_anomaly_points['value'], 
            color='green', label='Anomalies', s=50)
plt.title('LSTM Autoencoder Anomaly Detection')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)

# Plot 3: Original data with Ensemble anomalies
plt.subplot(3, 1, 3)
plt.plot(results['timestamp'], results['value'], label='Temperature')
ensemble_anomaly_points = results[results['ensemble_anomaly'] == 1]
plt.scatter(ensemble_anomaly_points['timestamp'], ensemble_anomaly_points['value'], 
            color='purple', label='Anomalies', s=50)
plt.title('Ensemble Approach Anomaly Detection')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('anomaly_detection_results.png')
plt.close()

# Save results to CSV
results.to_csv('anomaly_detection_results.csv', index=False)
print("Results saved to 'anomaly_detection_results.csv'")

print("\nANOMALY DETECTION TASK COMPLETED!")