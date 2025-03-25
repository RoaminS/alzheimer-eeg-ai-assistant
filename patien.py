# Patient simuler pour tester alz_simple.py

import numpy as np
import tensorflow as tf
from scipy.signal import welch, coherence
from statsmodels.tsa.arima_process import ArmaProcess

# 📌 Fonction CPU/GPU automatique
def setup_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✅ GPU détecté : {gpus[0].name}, utilisation activée")
            return "GPU"
        except RuntimeError as e:
            print(f"⚠️ Erreur GPU : {e}, basculement sur CPU")
            return "CPU"
    else:
        print("✅ Aucun GPU détecté, utilisation du CPU")
        return "CPU"

# 📌 Paramètres EEG
fs = 512
num_electrodes = 19
num_features = 5
samples = 768

# 📌 Définition des moyennes EEG
states = {
    "Sain": [0.8, 0.5, 1.2, 0.7, 0.4],
    "Début Alzheimer": [1.5, 0.8, 0.8, 0.5, 0.3],
    "Modéré Alzheimer": [2.0, 1.0, 0.5, 0.3, 0.2],
    "Avancé Alzheimer": [2.5, 1.2, 0.3, 0.1, 0.05]
}
std_dev = [0.3, 0.2, 0.3, 0.2, 0.1]

# 📌 Paramètres ARIMA
ar_params = {
    "Delta": [1, -0.7, 0.2],
    "Theta": [1, -0.6, 0.3],
    "Alpha": [1, -0.5, 0.4],
    "Beta": [1, -0.4, 0.5],
    "Gamma": [1, -0.3, 0.6]
}
ma_params = [1, 0.5]

# 📌 Génération des séries temporelles EEG
def generate_arima_eeg(mean, std, ar, ma, samples=768):
    arima_process = ArmaProcess(np.array(ar), np.array(ma)).generate_sample(nsample=samples)
    return mean + std * arima_process

# 📌 Filtre de Kalman
def kalman_filter(signal, process_noise=0.01, measurement_noise=0.1):
    x = signal[0]
    P = 1.0
    Q = process_noise
    R = measurement_noise
    filtered = []
    for z in signal:
        x_pred = x
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        filtered.append(x)
    return np.array(filtered)

# 🔹 Configurer le device
device = setup_device()

# 📌 Charger le modèle CNN-LSTM
model_file = "/root/eeg_alzheimer_final.keras"  # Ajuste si .h5
clf = tf.keras.models.load_model(model_file)
print(f"✅ Modèle CNN-LSTM chargé depuis {model_file}")

# 📌 Simuler un patient
patient_state = "Modéré Alzheimer"  # Ajustable
print(f"🚀 Simulation d’un patient avec état : {patient_state}")

eeg_data = np.zeros((num_electrodes, num_features, samples))
for i in range(num_electrodes):
    for j, band in enumerate(["Delta", "Theta", "Alpha", "Beta", "Gamma"]):
        raw_signal = generate_arima_eeg(states[patient_state][j], std_dev[j], ar_params[band], ma_params)
        noisy_signal = raw_signal + np.random.normal(0, 1.0, samples)
        eeg_data[i, j, :] = kalman_filter(noisy_signal)

if np.random.rand() < 0.3:
    artifact_duration = int(0.2 * fs)
    start = np.random.randint(0, samples - artifact_duration)
    eeg_data[:, :, start:start + artifact_duration] += np.sin(2 * np.pi * 50 * np.linspace(0, 0.2, artifact_duration))
    if np.random.rand() < 0.5:
        eeg_data[:5, :, start:start + artifact_duration] += np.exp(np.random.normal(0, 1, (5, num_features, artifact_duration)))
    eeg_data[:, 4, start:start + artifact_duration] += np.random.normal(0, 2, (num_electrodes, artifact_duration))

# Reshape pour CNN-LSTM
new_patient_eeg = eeg_data.transpose(2, 0, 1).reshape(1, samples, num_electrodes * num_features)

# 📌 Prédiction
prediction = clf.predict(new_patient_eeg)
pred_class = np.argmax(prediction, axis=1)[0]
class_names = ["Sain", "Début Alzheimer", "Modéré Alzheimer", "Avancé Alzheimer"]
print(f"🚀 L’IA prédit : {class_names[pred_class]} (Sain, Début Alzheimer, Modéré Alzheimer, Avancé Alzheimer)")
print(f"🚀 Probabilités : {prediction[0]}")  # Ajout des probas

# 📌 Stimulation recommandée
stimulation_recommendations = {
    0: "Aucune stimulation nécessaire.",
    1: "10 min de VR Alpha pour synchronisation.",
    2: "15 min de VR Theta + sommeil optimisé.",
    3: "20 min de VR Delta + consultation urgente."
}
print(f"🔹 Recommandation : {stimulation_recommendations[pred_class]}")
