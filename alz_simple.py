"""
alz_simple.py – Modèle IA simplifié de détection Alzheimer via EEG simulé

Auteur : Kocupyr Romain
Licence : Creative Commons BY-NC-SA 4.0
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from scipy.signal import welch, coherence
from statsmodels.tsa.arima_process import ArmaProcess
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

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

# 🔹 Charger ou initialiser
data_file = "/root/eeg_data.pkl"
model_file = "/root/eeg_alzheimer_final.keras"
max_samples = 10_000_000

if os.path.exists(data_file):
    X_loaded, y_loaded = joblib.load(data_file)
    print(f"✅ Chargement de {len(X_loaded)} échantillons existants")
else:
    X_loaded, y_loaded = [], []
    print("✅ Aucun fichier de données existant, démarrage à zéro")

if os.path.exists(model_file):
    clf = tf.keras.models.load_model(model_file)
    print("✅ Modèle CNN-LSTM chargé depuis", model_file)
else:
    clf = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(samples, num_electrodes * num_features)),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=False),
        Dense(32, activation='relu'),
        Dropout(0.7),
        Dense(4, activation='softmax')
    ])
    clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("✅ Nouveau modèle CNN-LSTM initialisé")

# 🔹 Générer de nouvelles données
num_samples = 10000
X_new, y_new = [], []

alpha_sain, beta_sain = 80, 20
p_sain = np.random.beta(alpha_sain, beta_sain)
print(f"✅ Proportion 'Sain' tirée : {p_sain * 100:.2f}%")

remaining_prob = 1 - p_sain
total_ratio = 10 + 7 + 3
p_debut = remaining_prob * (10 / total_ratio)
p_modere = remaining_prob * (7 / total_ratio)
p_avance = remaining_prob * (3 / total_ratio)

for _ in range(num_samples):
    progression = np.random.choice(
        ["Sain", "Début Alzheimer", "Modéré Alzheimer", "Avancé Alzheimer"],
        p=[p_sain, p_debut, p_modere, p_avance]
    )
    
    eeg_data = np.zeros((num_electrodes, num_features, samples))
    for i in range(num_electrodes):
        for j, band in enumerate(["Delta", "Theta", "Alpha", "Beta", "Gamma"]):
            raw_signal = generate_arima_eeg(states[progression][j], std_dev[j], ar_params[band], ma_params)
            noisy_signal = raw_signal + np.random.normal(0, 1.0, samples)
            eeg_data[i, j, :] = kalman_filter(noisy_signal)

    if np.random.rand() < 0.3:
        artifact_duration = int(0.2 * fs)
        start = np.random.randint(0, samples - artifact_duration)
        eeg_data[:, :, start:start + artifact_duration] += np.sin(2 * np.pi * 50 * np.linspace(0, 0.2, artifact_duration))
        if np.random.rand() < 0.5:
            eeg_data[:5, :, start:start + artifact_duration] += np.exp(np.random.normal(0, 1, (5, num_features, artifact_duration)))
        eeg_data[:, 4, start:start + artifact_duration] += np.random.normal(0, 2, (num_electrodes, artifact_duration))

    final_features = eeg_data.transpose(2, 0, 1).reshape(samples, num_electrodes * num_features)
    X_new.append(final_features)
    y_new.append(["Sain", "Début Alzheimer", "Modéré Alzheimer", "Avancé Alzheimer"].index(progression))

X = X_loaded + X_new
y = y_loaded + y_new
if len(X) > max_samples:
    X = X[-max_samples:]
    y = y[-max_samples:]
X = np.array(X)
y = np.array(y)
print(f"✅ Total d’échantillons après ajout (limité à {max_samples}) : {len(X)}")

# 📌 Sauvegarder les données
joblib.dump((X, y), data_file)
print(f"✅ Données sauvegardées sous {data_file}")

# 📌 Préparer les labels
y_onehot = to_categorical(y, num_classes=4)

# 📌 Générateur pour limiter la mémoire
def data_generator(X, y, batch_size=32):
    while True:
        idx = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch_idx = idx[start:start + batch_size]
            yield X[batch_idx], y[batch_idx]

# 📌 Séparer pour entraînement
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)
train_gen = data_generator(X_train, y_train, batch_size=32)
steps_per_epoch = len(X_train) // 32

# 📌 Entraînement
clf.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=3, validation_data=(X_test, y_test), verbose=1)

# 📌 Évaluation
y_pred = clf.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"✅ Précision finale : {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print("\n📊 Matrice de confusion :\n", conf_matrix)
print("\n📊 Rapport de classification :\n", classification_report(y_test_classes, y_pred_classes))
f1_macro = f1_score(y_test_classes, y_pred_classes, average='macro')
print(f"✅ Score F1 macro : {f1_macro * 100:.2f}%")

# 📌 Stimulation MemorAI
stimulation_recommendations = {
    0: "Aucune stimulation nécessaire.",
    1: "10 min de VR Alpha pour synchronisation.",
    2: "15 min de VR Theta + sommeil optimisé.",
    3: "20 min de VR Delta + consultation urgente."
}
class_names = ["Sain", "Début Alzheimer", "Modéré Alzheimer", "Avancé Alzheimer"]
for pred in set(y_pred_classes):
    print(f"🔹 Pour {class_names[pred]} : {stimulation_recommendations[pred]}")

# 📌 Visualisation
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, cmap="Blues")
plt.title("Matrice de confusion Alzheimer")
plt.colorbar()
plt.xlabel("Prédictions")
plt.ylabel("Vraies classes")
plt.xticks(ticks=[0, 1, 2, 3], labels=["Sain", "Début", "Modéré", "Avancé"])
plt.yticks(ticks=[0, 1, 2, 3], labels=["Sain", "Début", "Modéré", "Avancé"])
plt.savefig("/root/alzheimer_confusion_matrix.png")
print("✅ Matrice de confusion sauvegardée sous /root/alzheimer_confusion_matrix.png")

# 📌 Sauvegarde du modèle
clf.save(model_file)
print("✅ Modèle sauvegardé sous", model_file)


# ------------------------------------------------------------------------------

#📄 LICENCE - Creative Commons Attribution-NonCommercial-ShareAlike 4.0
#
# Ce script "alz_simple.py" fait partie du projet Alzheimer EEG AI Assistant,
# développé par Kocupyr Romain (romainsantoli@gmail.com).
#
# Vous êtes libres de :
# ✅ Partager — copier et redistribuer le script
# ✅ Adapter — le modifier, transformer et l’intégrer dans un autre projet
#
# Sous les conditions suivantes :
# 📌 Attribution — Vous devez mentionner l’auteur original (Kocupyr Romain)
# 📌 Non Commercial — Interdiction d’usage commercial sans autorisation
# 📌 Partage identique — Toute version modifiée doit être publiée sous la même licence
#
# 🔗 Licence complète : https://creativecommons.org/licenses/by-nc-sa/4.0/

# ------------------------------------------------------------------------------


