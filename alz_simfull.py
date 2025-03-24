"""
alz_reel.py – Modèle IA simplifié de détection Alzheimer via EEG simulé

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
import concurrent.futures

from scipy.signal import welch
from scipy.signal import stft

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

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

import tf2onnx

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

# 📊 États EEG par classe
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

# 🔁 Génération d’un signal EEG ARIMA
def generate_arima_eeg(mean, std, ar, ma, samples=768):
    arima_process = ArmaProcess(np.array(ar), np.array(ma)).generate_sample(nsample=samples)
    return mean + std * arima_process


# 🔧 UKF (Kalman ;) )
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

def ukf_physical_adaptive_batch(batch_data, fs=500, process_noise_init=0.1, measurement_noise_init=0.1):
    batch_size, n_channels, n_samples = batch_data.shape
    dt = 1 / fs
    out = np.zeros_like(batch_data)

    for b in range(batch_size):
        eeg_data = batch_data[b]
        points = MerweScaledSigmaPoints(n=n_channels, alpha=0.3, beta=2.0, kappa=1.0)
        ukf = UnscentedKalmanFilter(
            dim_x=n_channels, dim_z=n_channels, dt=dt,
            fx=lambda x, dt_local: physical_eeg_model(x, dt_local, freqs_window, coupling_window, band_powers_window),
            hx=measurement_function, points=points
        )
        ukf.x = eeg_data[:, 0].copy()
        ukf.P = np.eye(n_channels) * process_noise_init
        window_init = eeg_data[:, :min(100, n_samples)]
        Q_base = np.cov(window_init) if window_init.shape[1] > 1 else np.eye(n_channels) * process_noise_init
        R_base = np.eye(n_channels) * measurement_noise_init
        ukf.Q = Q_base.copy()
        ukf.R = R_base.copy()
        out[b, :, 0] = ukf.x

        min_window, max_window = 50, 200
        min_coupling, max_coupling = 0.05, 0.2
        window_size = 100
        coupling_window = 0.1
        freqs_window = np.ones(n_channels) * 10
        band_powers_window = np.ones((n_channels, 4))

        for t in range(1, n_samples):
            z = eeg_data[:, t]
            window_start = max(0, t - window_size)
            window_data = eeg_data[:, window_start:t]

            if window_data.shape[1] >= 10:
                try:
                    stability, freqs_window, band_powers_window = compute_spectral_features(window_data, fs, window_size)
                except:
                    stability = 0.5
                    freqs_window = np.ones(n_channels) * 10
                    band_powers_window = np.ones((n_channels, 4))

                window_size = int(min_window + (max_window - min_window) * stability)
                window_size = max(min_window, min(window_size, t))

                try:
                    corr_matrix = np.corrcoef(window_data)
                    off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                    correlation = np.mean(np.abs(off_diag))
                except:
                    correlation = 0.5
                coupling_window = min_coupling + (max_coupling - min_coupling) * correlation

                try:
                    cov = np.cov(window_data)
                    if not np.any(np.isnan(cov)) and not np.any(np.isinf(cov)):
                        ukf.Q = cov * process_noise_init
                except:
                    pass

                std_window = np.std(window_data, axis=1)
                std_window = np.where(std_window == 0, 1e-8, std_window)
                z_diff = np.abs(z - ukf.x)
                artifact_factor = np.where(z_diff > 3 * std_window, 10.0, 1.0)
                ukf.R = R_base * artifact_factor[:, np.newaxis]

            ukf.predict()
            ukf.update(z)

            if np.any(np.isnan(ukf.x)) or np.any(np.isinf(ukf.x)):
                print(f"⚠️ NaN/Inf UKF au timestep {t}, batch {b}")
                out[b] = np.zeros_like(eeg_data)
                break

            out[b, :, t] = ukf.x

    return out


# 🔁 Générateur complet de données EEG simulées avancées
def generate_advanced_simulation(num_samples, samples, num_electrodes, num_features, fs, states, std_dev, ar_params, ma_params):
    X_sim, y_sim, pids_sim = [], [], []
    alpha_sain, beta_sain = 80, 20
    p_sain = np.random.beta(alpha_sain, beta_sain)
    remaining_prob = 1 - p_sain
    total_ratio = 10 + 7 + 3
    p_debut = remaining_prob * (10 / total_ratio)
    p_modere = remaining_prob * (7 / total_ratio)
    p_avance = remaining_prob * (3 / total_ratio)
    print(f"✅ Proportion 'Sain' tirée : {p_sain * 100:.2f}%")

    for i in range(num_samples):
        progression = np.random.choice(
            ["Sain", "Début Alzheimer", "Modéré Alzheimer", "Avancé Alzheimer"],
            p=[p_sain, p_debut, p_modere, p_avance]
        )

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

            # ✅ UKF physique multi-électrodes par bande
            filtered_band = ukf_physical_adaptive_batch(band_matrix[np.newaxis, :, :], fs=fs)[0]
            eeg_data[:, j, :] = filtered_band

        # 🧨 Ajout artefacts
        if np.random.rand() < 0.3:
            artifact_duration = int(0.2 * fs)
            start = np.random.randint(0, samples - artifact_duration)
            eeg_data[:, :, start:start + artifact_duration] += np.sin(2 * np.pi * 50 * np.linspace(0, 0.2, artifact_duration))
            if np.random.rand() < 0.5:
                eeg_data[:5, :, start:start + artifact_duration] += np.exp(np.random.normal(0, 1, (5, num_features, artifact_duration)))
            eeg_data[:, 4, start:start + artifact_duration] += np.random.normal(0, 2, (num_electrodes, artifact_duration))

        final_features = eeg_data.transpose(2, 0, 1).reshape(samples, num_electrodes * num_features)
        final_features = normalize_eeg(final_features)

        if not np.any(np.isnan(final_features)) and not np.any(np.isinf(final_features)):
            X_sim.append(final_features)
            y_sim.append(["Sain", "Début Alzheimer", "Modéré Alzheimer", "Avancé Alzheimer"].index(progression))
            pids_sim.append(f"sim-{progression.lower()}-{i:06d}")
        else:
            print(f"⚠️ Segment simulé {i} ignoré (NaN/Inf détecté)")

        if i > 0 and i % 500 == 0:
            print(f"🧪 {i}/{num_samples} segments simulés...")

    return np.array(X_sim), np.array(y_sim), np.array(pids_sim)



# 🔁 Génération simulée avancée intégrée
X_sim, y_sim, pids_sim = generate_advanced_simulation(
    num_samples=1000, #10 000 sinon
    samples=samples,
    num_electrodes=num_electrodes,
    num_features=num_features,
    fs=fs,
    states=states,
    std_dev=std_dev,
    ar_params=ar_params,
    ma_params=ma_params
)

# 🔧 Fusion avec les données (si existantes)
if os.path.exists(data_file):
    X_loaded, y_loaded, pids_loaded = joblib.load(data_file)
    print(f"✅ Chargement de {len(X_loaded)} segments existants")
    X = np.concatenate([X_loaded, X_sim])
    y = np.concatenate([y_loaded, y_sim])
    patient_ids = np.concatenate([pids_loaded, pids_sim])
else:
    print("✅ Pas de données sauvegardées, on part des données simulées")
    X = X_sim
    y = y_sim
    patient_ids = pids_sim

# 🔧 Déduplication + équilibrage avec FAISS
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
    X_unique = X[unique_mask]
    y_unique = y[unique_mask]
    pids_unique = patient_ids[unique_mask]
    print(f"✅ Déduplication : {len(X)} → {len(X_unique)} uniques")

    X_bal, y_bal, pids_bal = [], [], []
    counter = Counter(y_unique)
    for cls in range(max(counter.keys()) + 1):
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
        X_final = X_final[idx]
        y_final = y_final[idx]
        pids_final = pids_final[idx]

    faiss.write_index(index, faiss_file)
    joblib.dump((y_final, pids_final), metadata_file)
    print(f"✅ Données FAISS équilibrées : {len(X_final)}")
    return X_final, y_final, pids_final

X, y, patient_ids = faiss_deduplicate_and_balance(
    X, y, patient_ids,
    max_samples=max_samples,
    target_per_class=20000,
    faiss_file=faiss_file,
    metadata_file=metadata_file
)

# 🔒 Sauvegarde
joblib.dump((X, y, patient_ids), data_file)
print(f"✅ Données sauvegardées dans : {data_file}")


# 🔁 Encodage one-hot
y_cat = to_categorical(y, num_classes=num_classes)

# 🔁 Split stratifié par patient
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_cat[train_idx], y_cat[test_idx]
pids_train, pids_test = patient_ids[train_idx], patient_ids[test_idx]

# 🔁 Split Val
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx2, val_idx = next(gss_val.split(X_train, y_train, groups=pids_train))
X_train_final, X_val = X_train[train_idx2], X_train[val_idx]
y_train_final, y_val = y_train[train_idx2], y_train[val_idx]
pids_train_final, pids_val = pids_train[train_idx2], pids_train[val_idx]

print(f"✅ Train: {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 🔧 Bloc Attention
def attention_block(inputs, name_suffix=""):
    avg_pool = GlobalAveragePooling1D()(inputs)
    att = Dense(inputs.shape[1], activation='softmax', name=f"att_dense_{name_suffix}")(avg_pool)
    att = Reshape((inputs.shape[1], 1))(att)
    out = Multiply()([inputs, att])
    return out

# 🔧 Construction du modèle
input_layer = Input(shape=(samples, num_electrodes * num_features))
x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Bidirectional(LSTM(50, return_sequences=True))(x)
x = BatchNormalization()(x)
x = attention_block(x, "alz")
x = Bidirectional(LSTM(32))(x)
x = BatchNormalization()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.7)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

clf = Model(inputs=input_layer, outputs=output_layer)
clf.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CategoricalFocalCrossentropy(gamma=6),
    metrics=['accuracy']
)

# 🔧 Callbacks
checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# 🔁 Entraînement
batch_size = 32
steps_per_epoch = len(X_train_final) // batch_size

def data_generator(X, y, batch_size):
    while True:
        idx = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            yield X[idx[start:end]], y[idx[start:end]]

train_gen = data_generator(X_train_final, y_train_final, batch_size)

class_weights = dict(enumerate(compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train_final, axis=1)),
    y=np.argmax(y_train_final, axis=1)
)))
print(f"✅ Poids de classes : {class_weights}")

clf.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=5, # (normalement à 30, mais 5 pour les tests)
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# 🔁 Évaluation
clf = load_model(model_file, compile=False)
clf.compile(optimizer=Adam(learning_rate=0.001), loss=CategoricalFocalCrossentropy(gamma=6), metrics=['accuracy'])

y_pred = clf.predict(X_test)
y_true_cls = np.argmax(y_test, axis=1)
y_pred_cls = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_true_cls, y_pred_cls)
f1 = f1_score(y_true_cls, y_pred_cls, average='macro')
cm = confusion_matrix(y_true_cls, y_pred_cls)
report = classification_report(y_true_cls, y_pred_cls)

print(f"✅ Précision : {acc*100:.2f}%")
print(f"✅ F1 macro : {f1*100:.2f}%")
print("📊 Matrice de confusion :\n", cm)
print("📊 Rapport de classification :\n", report)

# 🔁 Export ONNX
model_proto, _ = tf2onnx.convert.from_keras(clf, output_path=onnx_file)
print(f"✅ Export ONNX → {onnx_file}")

# 🔁 Export TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(clf)
tflite_model = converter.convert()
with open(tflite_file, "wb") as f:
    f.write(tflite_model)
print(f"✅ Export TFLite → {tflite_file}")



 
# ------------------------------------------------------------------------------
# 📄 LICENCE - Creative Commons Attribution-NonCommercial-ShareAlike 4.0
#
# Ce script "alz_reel.py" fait partie du projet Alzheimer EEG AI Assistant,
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
