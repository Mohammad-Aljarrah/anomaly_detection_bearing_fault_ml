import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import OneClassSVM

# Step1: loading data
folder = "C:/Users/k67192ma/PycharmProjects/anomaly_detection_bearing_fault_ml/data/4.+Bearings/4. Bearings/IMS/1st_test/1st_test"
files = sorted([f for f in os.listdir(folder) if f.endswith(".03") or f.endswith(".07")  \
                or f.endswith(".13" )  or f.endswith(".17")  or f.endswith(".24")  or f.endswith(".30")\
                or f.endswith(".32" )  or f.endswith(".36")  or f.endswith(".44")  or f.endswith(".46")\
                or f.endswith(".56" )  or f.endswith(".58") ])


all_data = []
for f in files:
    path = os.path.join(folder, f)
    data = np.loadtxt(path)  # ASCII text
    print(f"{f}: shape {data.shape}")
    all_data.append(data)

print(f"Total files loaded: {len(all_data)}")

#Step2: Vibration signals are long (tens of thousands of samples)
#ML algorithms work on fixed-size feature vectors
#We split signals into overlapping windows

#Recommended parameters
#Window length: 2048 samples
#Overlap: 50% → 1024 samples
def window_signal(signal, win_len=2048, overlap=1024):
    step = win_len - overlap
    windows = []
    for start in range(0, len(signal) - win_len + 1, step):
        windows.append(signal[start:start + win_len])
    return np.array(windows)


#Step3: Feature extraction
#Convert each window into numerical features

#Simple time-domain features for a start:
#Mean
#Standard deviation
#RMS
#Max
#Min
#Peak-to-peak
def extract_features(windows):
    features = []
    for w in windows:
        features.append([
            np.mean(w),
            np.std(w),
            np.max(w),
            np.min(w),
            np.sqrt(np.mean(w**2)),  # RMS
            np.ptp(w)                # peak-to-peak
        ])
    return np.array(features)


#Step 4 — Apply to all channels
all_features = []
for ch in range(data.shape[1]):
    signal = data[:, ch]
    windows = window_signal(signal)
    feats = extract_features(windows)
    all_features.append(feats)

# Concatenate along columns
X = np.hstack(all_features)
print(X.shape)  # (num_windows, num_features * num_channels)

##### STEP 5 ########
############ ML-SVM Part #################
# 5.1 Data Splitting

# 5.2 Fit One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
# nu = fraction of expected outliers in training
oc_svm.fit(X_train)

# 5.3 Predict on test windows
y_pred = oc_svm.predict(X_test)
# -1 → anomaly, 1 → normal

#5.4 Visualize
# Convert predictions to 0/1 for plotting
y_plot = np.array([0 if y==-1 else 1 for y in y_pred])

plt.figure(figsize=(12,4))
plt.plot(y_plot, marker='o', linestyle='')
plt.title("Anomaly detection results (1=normal, 0=anomaly)")
plt.xlabel("Window index")
plt.ylabel("Prediction")
plt.show()

# Example: first channel of first file

plt.plot(all_data[0][:, 0]  )
plt.title("Sample Vibration Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.show()