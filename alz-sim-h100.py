"""
alz-sim-h100.py – Modèle IA simplifié de détection Alzheimer via EEG simulé

Auteur : Kocupyr Romain
Licence : Creative Commons BY-NC-SA 4.0
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
import faiss
import gc
import multiprocessing
import concurrent.futures

from scipy.signal import welch, stft
from statsmodels.tsa.arima_process import ArmaProcess
from collections import Counter, defaultdict

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
                                     Bidirectional, BatchNormalization, Multiply, GlobalAveragePooling1D,
                                     Reshape)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

import tf2onnx

# 🔧 Détection auto des CPUs
NUM_CORES = multiprocessing.cpu_count()
print(f"✅ CPUs détectés : {NUM_CORES}")

# 🔧 GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20480)])
        print(f"✅ GPU configuré : {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️ Erreur GPU : {e}, fallback CPU")
else:
    print("✅ Aucun GPU détecté, utilisation CPU")

# 📌 PARAMÈTRES GLOBAUX
fs = 500
samples = 768
num_electrodes = 19
num_features = 5
num_classes = 4
max_samples = 10_000_000
data_dir = "/workspace/memory_os_ai/alz/"
faiss_file = os.path.join(data_dir, "faiss_index.bin")
metadata_file = os.path.join(data_dir, "faiss_metadata.pkl")
data_file = os.path.join(data_dir, "eeg_data_alzheimer.pkl")
model_file = os.path.join(data_dir, "alz_model_alzheimer.keras")
onnx_file = os.path.join(data_dir, "alz_model_alzheimer.onnx")
tflite_file = os.path.join(data_dir, "alz_model_alzheimer.tflite")

states = {
    "Sain": [0.8, 0.5, 1.2, 0.7, 0.4],
    "Début Alzheimer": [1.5, 0.8, 0.8, 0.5, 0.3],
    "Modéré Alzheimer": [2.0, 1.0, 0.5, 0.3, 0.2],
    "Avancé Alzheimer": [2.5, 1.2, 0.3, 0.1, 0.05]
}
std_dev = [0.3, 0.2, 0.3, 0.2, 0.1]

ar_params = {
    "Delta": [1, -0.7, 0.2],
    "Theta": [1, -0.6, 0.3],
    "Alpha": [1, -0.5, 0.4],
    "Beta": [1, -0.4, 0.5],
    "Gamma": [1, -0.3, 0.6]
}
ma_params = [1, 0.5]

# 🔧 Normalisation EEG
def normalize_eeg(segment):
    segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.mean(segment, axis=0)
    std = np.std(segment, axis=0)
    std = np.where(std == 0, 1e-8, std)
    normalized = (segment - mean) / std
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

# 🔁 Génération signal EEG ARIMA
def generate_arima_eeg(mean, std, ar, ma, samples=768):
    arima_process = ArmaProcess(np.array(ar), np.array(ma)).generate_sample(nsample=samples)
    return mean + std * arima_process

# 🔧 Modèle EEG physique (UKF)
def physical_eeg_model(x, dt, freqs, coupling, band_powers):
    n_channels = len(x)
    dx = np.zeros(n_channels)
    for i in range(n_channels):
        omega = 2 * np.pi * freqs[i]
        alpha_power = band_powers[i, 2] / (np.sum(band_powers[i]) + 1e-8)
        amplitude_mod = np.clip(1 + alpha_power, 0.1, 10)
        self_term = -omega**2 * np.sin(omega * dt) * x[i] * amplitude_mod
        coupling_term = coupling * np.mean(x - x[i])
        dx[i] = self_term + coupling_term
    return x + dt * dx

def measurement_function(x):
    return x

# 🔧 UKF optimisé avec multi-threading adaptatif
def ukf_physical_adaptive_batch(batch_data, fs=500, process_noise_init=0.1, measurement_noise_init=0.1):
    batch_size, n_channels, n_samples = batch_data.shape
    dt = 1 / fs
    out = np.zeros_like(batch_data)

    def ukf_single_sample(eeg_data):
        points = MerweScaledSigmaPoints(n=n_channels, alpha=0.3, beta=2.0, kappa=1.0)
        ukf = UnscentedKalmanFilter(
            dim_x=n_channels, dim_z=n_channels, dt=dt,
            fx=lambda x, dt_local: physical_eeg_model(x, dt_local, freqs_window, coupling_window, band_powers_window),
            hx=measurement_function, points=points
        )
        ukf.x = eeg_data[:, 0].copy()
        ukf.P = np.eye(n_channels) * process_noise_init
        ukf.Q = np.eye(n_channels) * process_noise_init
        ukf.R = np.eye(n_channels) * measurement_noise_init
        result = np.zeros_like(eeg_data)

        for t in range(1, n_samples):
            z = eeg_data[:, t]
            ukf.predict()
            ukf.update(z)
            result[:, t] = ukf.x
        return result

    max_threads = min(NUM_CORES, batch_size)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(ukf_single_sample, batch_data[b]): b for b in range(batch_size)}
        for future in concurrent.futures.as_completed(futures):
            b = futures[future]
            result = future.result()
            if not np.any(np.isnan(result)) and not np.any(np.isinf(result)):
                out[b] = result
            else:
                print(f"⚠️ NaN/Inf UKF dans batch {b}")
                out[b] = np.zeros_like(batch_data[b])

    return out

# 🔁 Fonction de génération parallèle de segments EEG simulés
def generate_single_eeg_sample(i, states, std_dev, ar_params, ma_params, samples, num_electrodes, num_features, fs):
    progression = np.random.choice(["Sain", "Début Alzheimer", "Modéré Alzheimer", "Avancé Alzheimer"],
                                   p=[0.8, 0.1, 0.07, 0.03])

    eeg_data = np.zeros((num_electrodes, num_features, samples))

    for j, band in enumerate(["Delta", "Theta", "Alpha", "Beta", "Gamma"]):
        band_matrix = np.zeros((num_electrodes, samples))
        for ch in range(num_electrodes):
            raw_signal = generate_arima_eeg(
                mean=states[progression][j],
                std=std_dev[j],
                ar=ar_params[band],
                ma=ma_params,
                samples=samples
            )
            noisy_signal = raw_signal + np.random.normal(0, 1.0, samples)
            band_matrix[ch, :] = noisy_signal

        filtered_band = ukf_physical_adaptive_batch(band_matrix[np.newaxis, :, :], fs=fs)[0]
        eeg_data[:, j, :] = filtered_band

    if np.random.rand() < 0.3:
        artifact_duration = int(0.2 * fs)
        start = np.random.randint(0, samples - artifact_duration)
        eeg_data[:, :, start:start + artifact_duration] += np.sin(
            2 * np.pi * 50 * np.linspace(0, 0.2, artifact_duration))
        if np.random.rand() < 0.5:
            eeg_data[:5, :, start:start + artifact_duration] += np.exp(
                np.random.normal(0, 1, (5, num_features, artifact_duration)))
        eeg_data[:, 4, start:start + artifact_duration] += np.random.normal(
            0, 2, (num_electrodes, artifact_duration))

    final_features = eeg_data.transpose(2, 0, 1).reshape(samples, num_electrodes * num_features)
    final_features = normalize_eeg(final_features)

    if not np.any(np.isnan(final_features)) and not np.any(np.isinf(final_features)):
        return final_features, ["Sain", "Début Alzheimer", "Modéré Alzheimer", "Avancé Alzheimer"].index(progression), f"sim-{progression.lower()}-{i:06d}"
    else:
        print(f"⚠️ Segment {i} ignoré (NaN/Inf détecté)")
        return None

# 🚀 Génération parallèle adaptative
def generate_advanced_simulation_parallel(num_samples, samples, num_electrodes, num_features, fs, states, std_dev, ar_params, ma_params):
    X_sim, y_sim, pids_sim = [], [], []

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = {executor.submit(generate_single_eeg_sample, i, states, std_dev, ar_params, ma_params,
                                   samples, num_electrodes, num_features, fs): i for i in range(num_samples)}

        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result is not None:
                X_single, y_single, pid_single = result
                X_sim.append(X_single)
                y_sim.append(y_single)
                pids_sim.append(pid_single)

            if idx % 100 == 0 and idx > 0:
                print(f"🧪 {idx}/{num_samples} segments simulés en parallèle...")

    return np.array(X_sim), np.array(y_sim), np.array(pids_sim)

# 🔁 Appel de la génération parallèle
X_sim, y_sim, pids_sim = generate_advanced_simulation_parallel(
    num_samples=10000,  # Ajustable selon ressources disponibles
    samples=samples,
    num_electrodes=num_electrodes,
    num_features=num_features,
    fs=fs,
    states=states,
    std_dev=std_dev,
    ar_params=ar_params,
    ma_params=ma_params
)


# 🔧 Fusion données existantes si présentes
if os.path.exists(data_file):
    X_loaded, y_loaded, pids_loaded = joblib.load(data_file)
    print(f"✅ Chargement de {len(X_loaded)} segments existants")
    X = np.concatenate([X_loaded, X_sim])
    y = np.concatenate([y_loaded, y_sim])
    patient_ids = np.concatenate([pids_loaded, pids_sim])
else:
    print("✅ Pas de données sauvegardées, utilisation données simulées")
    X, y, patient_ids = X_sim, y_sim, pids_sim

# 🔧 FAISS déduplication & équilibrage
def create_or_load_faiss_index(dim, faiss_file):
    if os.path.exists(faiss_file):
        index = faiss.read_index(faiss_file)
        print(f"✅ Index FAISS chargé : {index.ntotal} vecteurs")
    else:
        index = faiss.IndexFlatL2(dim)
        print("✅ Nouvel index FAISS créé")
    return index

def faiss_deduplicate_and_balance(X, y, patient_ids, max_samples, target_per_class, faiss_file, metadata_file):
    X_flat = X.reshape(X.shape[0], -1).astype(np.float32)
    index = create_or_load_faiss_index(X_flat.shape[1], faiss_file)
    index.add(X_flat)
    D, I = index.search(X_flat, 2)
    unique_mask = D[:, 1] > 1e-6
    X_unique, y_unique, pids_unique = X[unique_mask], y[unique_mask], patient_ids[unique_mask]
    print(f"✅ Déduplication : {len(X)} → {len(X_unique)} segments uniques")

    X_bal, y_bal, pids_bal = [], [], []
    counter = Counter(y_unique)
    for cls in range(num_classes):
        idx = np.where(y_unique == cls)[0]
        n = min(target_per_class, len(idx))
        chosen = np.random.choice(idx, n, replace=False)
        X_bal.append(X_unique[chosen])
        y_bal.append(y_unique[chosen])
        pids_bal.append(pids_unique[chosen])

    X_final = np.concatenate(X_bal)
    y_final = np.concatenate(y_bal)
    pids_final = np.concatenate(pids_bal)

    if len(X_final) > max_samples:
        idx = np.random.choice(len(X_final), max_samples, replace=False)
        X_final, y_final, pids_final = X_final[idx], y_final[idx], pids_final[idx]

    faiss.write_index(index, faiss_file)
    joblib.dump((y_final, pids_final), metadata_file)
    print(f"✅ FAISS équilibrage terminé : {len(X_final)} segments finaux")
    return X_final, y_final, pids_final

X, y, patient_ids = faiss_deduplicate_and_balance(
    X, y, patient_ids, max_samples, 20000, faiss_file, metadata_file)

joblib.dump((X, y, patient_ids), data_file)
print(f"✅ Données sauvegardées dans : {data_file}")

# 🔁 Encodage one-hot
y_cat = to_categorical(y, num_classes=num_classes)

# 🔁 Splits stratifiés
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_cat[train_idx], y_cat[test_idx]
pids_train, pids_test = patient_ids[train_idx], patient_ids[test_idx]

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx2, val_idx = next(gss_val.split(X_train, y_train, groups=pids_train))
X_train_final, X_val = X_train[train_idx2], X_train[val_idx]
y_train_final, y_val = y_train[train_idx2], y_train[val_idx]

print(f"✅ Train: {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 🔧 Bloc Attention
def attention_block(inputs, name_suffix=""):
    avg_pool = GlobalAveragePooling1D()(inputs)
    att = Dense(inputs.shape[1], activation='softmax', name=f"att_dense_{name_suffix}")(avg_pool)
    att = Reshape((inputs.shape[1], 1))(att)
    return Multiply()([inputs, att])

# 🔧 Modèle CNN-BiLSTM avec Attention
input_layer = Input(shape=(samples, num_electrodes * num_features))
x = Conv1D(128, 3, activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Conv1D(256, 3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = BatchNormalization()(x)
x = attention_block(x, "alz")
x = Bidirectional(LSTM(64))(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.7)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

clf = Model(inputs=input_layer, outputs=output_layer)
clf.compile(Adam(1e-3), CategoricalFocalCrossentropy(gamma=6), ['accuracy'])

# 🔧 Callbacks
checkpoint = ModelCheckpoint(model_file, 'val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# 🔁 Entraînement adaptatif
batch_size = 32
steps_per_epoch = len(X_train_final) // batch_size

def data_generator(X, y, batch_size):
    while True:
        idx = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            yield X[idx[start:start + batch_size]], y[idx[start:start + batch_size]]

train_gen = data_generator(X_train_final, y_train_final, batch_size)

class_weights = dict(enumerate(compute_class_weight(
    'balanced', classes=np.unique(np.argmax(y_train_final, axis=1)),
    y=np.argmax(y_train_final, axis=1))))
print(f"✅ Poids des classes : {class_weights}")

clf.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=5,
        validation_data=(X_val, y_val), callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights, verbose=1)

# 🔁 Évaluation
clf = load_model(model_file, compile=False)
clf.compile(Adam(1e-3), CategoricalFocalCrossentropy(gamma=6), ['accuracy'])

y_pred = clf.predict(X_test)
y_true_cls, y_pred_cls = np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)

print(f"✅ Précision : {accuracy_score(y_true_cls, y_pred_cls)*100:.2f}%")
print(f"✅ F1 macro : {f1_score(y_true_cls, y_pred_cls, average='macro')*100:.2f}%")
print("📊 Matrice confusion :\n", confusion_matrix(y_true_cls, y_pred_cls))
print("📊 Rapport classification :\n", classification_report(y_true_cls, y_pred_cls))

# 🔁 Export ONNX & TFLite
tf2onnx.convert.from_keras(clf, output_path=onnx_file)
print(f"✅ Export ONNX → {onnx_file}")

tflite_model = tf.lite.TFLiteConverter.from_keras_model(clf).convert()
open(tflite_file, "wb").write(tflite_model)
print(f"✅ Export TFLite → {tflite_file}")




# ------------------------------------------------------------------------------
# 📄 LICENCE - Creative Commons Attribution-NonCommercial-ShareAlike 4.0
#
# Ce script "alz-sim-h100.py" fait partie du projet Alzheimer EEG AI Assistant,
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
