import os
import numpy as np
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import joblib

# ---------------------------
# User parameters
# ---------------------------
project_root = "C:/Users/k67192ma/PycharmProjects/anomaly_detection_bearing_fault_ml"
folder = os.path.join(project_root, "data", "4.+Bearings", "4. Bearings", "IMS", "1st_test", "1st_test")
num_healthy_files = 500  # first 500 files assumed healthy
window_length = 2048
overlap = 1024

# ---------------------------
# Step 1: List files
# ---------------------------
files = sorted([f for f in os.listdir(folder) if f.endswith(".03") or f.endswith(".07")  \
                or f.endswith(".13" )  or f.endswith(".17")  or f.endswith(".24")  or f.endswith(".30")\
                or f.endswith(".32" )  or f.endswith(".36")  or f.endswith(".44")  or f.endswith(".46")\
                or f.endswith(".56" )  or f.endswith(".58") ])
print(f"Total files found: {len(files)}")


# ---------------------------
# Step 2: Windowing function
# ---------------------------
def window_signal(signal, win_len=2048, overlap=1024):
    step = win_len - overlap
    windows = []
    for start in range(0, len(signal) - win_len + 1, step):
        windows.append(signal[start:start + win_len])
    return np.array(windows)


# ---------------------------
# Step 3: Feature extraction
# ---------------------------
def extract_features(windows):
    features = []
    for w in windows:
        features.append([
            np.mean(w),
            np.std(w),
            np.max(w),
            np.min(w),
            np.sqrt(np.mean(w ** 2)),  # RMS
            np.ptp(w)  # peak-to-peak
        ])
    return np.array(features)


# ---------------------------
# Step 4: Load files and prepare training/testing sets
# ---------------------------
X_train_list = []
X_test_list = []

for idx, f in enumerate(files):
    path = os.path.join(folder, f)
    data = np.loadtxt(path)  # ASCII text file, shape (samples, 8)

    # Window each channel and extract features
    channel_features = []
    for ch in range(data.shape[1]):  # automatically loop through all 8 channels
        windows = window_signal(data[:, ch], win_len=window_length, overlap=overlap)
        feats = extract_features(windows)
        channel_features.append(feats)

    # Concatenate features from all channels
    file_features = np.hstack(channel_features)

    # Assign to training or testing
    if idx < num_healthy_files:
        X_train_list.append(file_features)
    X_test_list.append(file_features)

# Combine all windows into final matrices
X_train = np.vstack(X_train_list)
X_test = np.vstack(X_test_list)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# ---------------------------
# Step 5: Train One-Class SVM
# ---------------------------
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
oc_svm.fit(X_train)

###########################################################
# Save the trained model
joblib.dump(oc_svm, "models/oc_svm_model.pkl")

# Later, load the model
#oc_svm_loaded = joblib.load("models/oc_svm_model.pkl")

# Use it for prediction
#y_pred = oc_svm_loaded.predict(X_test)


########################################################
# ---------------------------
# Step 6: Predict anomalies
# ---------------------------
y_pred = oc_svm.predict(X_test)  # -1=anomaly, 1=normal
y_plot = np.array([0 if y == -1 else 1 for y in y_pred])

# ---------------------------
# Step 7: Visualize
# ---------------------------
plt.figure(figsize=(12, 4))
plt.plot(y_plot, marker='o', linestyle='', markersize=2)
plt.title("Anomaly detection results (1=normal, 0=anomaly)")
plt.xlabel("Window index")
plt.ylabel("Prediction")
plt.show()




############## single file testung
filename ="C:/Users/k67192ma/PycharmProjects/anomaly_detection_bearing_fault_ml/data/4.+Bearings/4. Bearings/IMS/1st_test/1st_test/2003.11.23.12.46.56"
data = np.loadtxt(filename)  # shape = (20480, 8)

# Window & extract features for all channels
channel_features = []
for ch in range(data.shape[1]):
    windows = window_signal(data[:, ch], win_len=window_length, overlap=overlap)
    feats = extract_features(windows)
    channel_features.append(feats)

file_features = np.hstack(channel_features)

# ---------------------------
# Predict with trained SVM
# ---------------------------
# Assume oc_svm is already trained on healthy windows
y_pred = oc_svm.predict(file_features)  # -1=anomaly, 1=normal
y_plot = np.array([0 if y==-1 else 1 for y in y_pred])

# ---------------------------
# Plot results
# ---------------------------
plt.figure(figsize=(12,4))
plt.plot(y_plot, marker='o', linestyle='', markersize=3)
plt.title(f"Anomaly detection for {filename}")
plt.xlabel("Window index")
plt.ylabel("Prediction (1=normal, 0=anomaly)")
plt.show()
