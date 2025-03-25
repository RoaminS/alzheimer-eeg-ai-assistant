 """
alz-reel-ad.py – Modèle IA simplifié de détection Alzheimer via EEG simulé

Auteur : Kocupyr Romain
Licence : Creative Commons BY-NC-SA 4.0
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""    

import numpy as np
import mne
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import faiss
import multiprocessing
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.signal import stft
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
                                     Bidirectional, BatchNormalization, Multiply, GlobalAveragePooling1D, 
                                     Reshape)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict, Counter
import joblib
import tf2onnx
from statsmodels.tsa.arima_process import ArmaProcess
import seaborn as sns

# =============================================================================
#                      >>>  Fonctions & Configuration  <<<
# =============================================================================

def setup_device_and_faiss():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✅ GPU détecté : {gpus[0].name}, tentative d'utilisation FAISS-GPU")
            try:
                res = faiss.StandardGpuResources()
                print("✅ FAISS-GPU activé")
                return "GPU", res
            except AttributeError:
                print("⚠️ FAISS-GPU non disponible, basculement sur FAISS-CPU avec GPU TensorFlow")
                faiss.omp_set_num_threads(multiprocessing.cpu_count())
                return "GPU", None
        except RuntimeError as e:
            print(f"⚠️ Erreur GPU : {e}, basculement sur CPU")
            faiss.omp_set_num_threads(multiprocessing.cpu_count())
            return "CPU", None
    else:
        print("✅ Aucun GPU détecté, utilisation de FAISS-CPU optimisé")
        faiss.omp_set_num_threads(multiprocessing.cpu_count())
        return "CPU", None

def create_or_load_faiss_index(dim, faiss_file, use_gpu=False, gpu_resources=None):
    if os.path.exists(faiss_file):
        index = faiss.read_index(faiss_file)
        print(f"✅ Index FAISS chargé depuis {faiss_file} ({index.ntotal} éléments)")
    else:
        index = faiss.IndexFlatL2(dim)
        print(f"✅ Nouvel index FAISS FlatL2 créé (précision maximale)")
    if use_gpu and gpu_resources is not None:
        index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
        print("✅ Index déplacé sur GPU")
    return index

def sanitize_eeg(eeg_data, verbose=False):
    """Nettoie les NaN/infs et gère les segments plats."""
    eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
    eeg_data = np.clip(eeg_data, -1e6, 1e6)
    stds = np.std(eeg_data, axis=1)
    too_flat = stds < 1e-10
    if np.any(too_flat):
        if verbose:
            print(f"⚠️ Segments plats détectés (std < 1e-10) dans {np.where(too_flat)[0]}, ajout de bruit léger")
        for i in np.where(too_flat)[0]:
            eeg_data[i] += np.random.normal(0, 1e-6, eeg_data.shape[1])
    return eeg_data, too_flat

def compute_spectral_features(data, fs, window_size):
    data, _ = sanitize_eeg(data, verbose=True)
    if data.shape[1] < window_size:
        freqs, _, Zxx = stft(data, fs=fs, nperseg=data.shape[1], noverlap=data.shape[1]//2)
    else:
        freqs, _, Zxx = stft(data, fs=fs, nperseg=window_size, noverlap=window_size//2)
    
    power = np.mean(np.abs(Zxx)**2, axis=2)
    if np.any(np.isnan(power)) or np.any(np.isinf(power)):
        power, _ = sanitize_eeg(power, verbose=True)
    dominant_freqs = freqs[np.argmax(power, axis=1)]
    
    bands = {'delta': (0, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}
    band_powers = np.zeros((data.shape[0], len(bands)))
    for i, (band, (fmin, fmax)) in enumerate(bands.items()):
        idx = (freqs >= fmin) & (freqs <= fmax)
        band_powers[:, i] = np.mean(power[:, idx], axis=1)
    if np.any(np.isnan(band_powers)) or np.any(np.isinf(band_powers)):
        band_powers, _ = sanitize_eeg(band_powers, verbose=True)
    
    if Zxx.shape[2] > 1:
        freqs_over_time = freqs[np.argmax(np.abs(Zxx)**2, axis=1)]
        variance = np.mean(np.var(freqs_over_time, axis=1))
        if np.isnan(variance) or np.isinf(variance):
            variance = 0
        stability = np.clip(1 / (1 + variance), 0, 1)
    else:
        stability = 0.5
    
    return stability, dominant_freqs, band_powers

def compute_inter_channel_correlation(data):
    data, _ = sanitize_eeg(data, verbose=True)
    if data.shape[1] < 2:
        return 0.1
    corr_matrix = np.corrcoef(data)
    if np.any(np.isnan(corr_matrix)) or np.any(np.isinf(corr_matrix)):
        return 0.1
    off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    return np.mean(np.abs(off_diag))

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

def ukf_physical_adaptive(eeg_data, fs=500, process_noise_init=0.1, measurement_noise_init=0.1):
    eeg_data, _ = sanitize_eeg(eeg_data, verbose=True)
    n_channels, n_samples = eeg_data.shape
    dt = 1 / fs

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
    if np.any(np.isnan(Q_base)) or np.any(np.isinf(Q_base)):
        Q_base = np.eye(n_channels) * process_noise_init
    R_base = np.eye(n_channels) * measurement_noise_init
    
    ukf.Q = Q_base.copy()
    ukf.R = R_base.copy()

    out = np.zeros_like(eeg_data)
    out[:, 0] = ukf.x

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
            stability, freqs_window, band_powers_window = compute_spectral_features(window_data, fs, window_size)
            window_size = int(min_window + (max_window - min_window) * stability)
            window_size = max(min_window, min(window_size, t))
            correlation = compute_inter_channel_correlation(window_data)
            coupling_window = min_coupling + (max_coupling - min_coupling) * correlation
            cov = np.cov(window_data)
            if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
                cov = np.eye(n_channels) * process_noise_init
            ukf.Q = cov * process_noise_init
            z_diff = np.abs(z - ukf.x)
            std_window = np.std(window_data, axis=1)
            std_window = np.where(std_window < 1e-8, 1e-8, std_window)
            artifact_factor = np.where(z_diff > 3 * std_window, 10.0, 1.0)
            ukf.R = R_base * artifact_factor[:, np.newaxis]
        
        ukf.predict()
        ukf.update(z)
        if np.any(np.isnan(ukf.x)) or np.any(np.isinf(ukf.x)):
            print(f"⚠️ NaN/Inf dans ukf.x au timestep {t}, réinitialisation")
            ukf.x, _ = sanitize_eeg(ukf.x)
        out[:, t] = ukf.x

    return out

def normalize_eeg(segment):
    segment, _ = sanitize_eeg(segment, verbose=True)
    mean = np.mean(segment, axis=0)
    std = np.std(segment, axis=0)
    std = np.where(std < 1e-8, 1e-8, std)
    return (segment - mean) / std

def augment_eeg(segment):
    shift = np.random.randint(-5, 5)
    segment = np.roll(segment, shift, axis=0)
    noise = np.random.normal(0, 0.01, segment.shape)
    segment += noise
    scale = np.random.uniform(0.98, 1.02)
    segment *= scale
    return segment

def generate_arima_eeg(mean, std, ar, ma, samples=768):
    arima_process = ArmaProcess(np.array(ar), np.array(ma)).generate_sample(nsample=samples)
    return mean + std * arima_process

def data_generator(X, y, groups, batch_size=64, augment_rare=True):
    while True:
        idx = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch_idx = idx[start:start + batch_size]
            X_batch = X[batch_idx].copy()
            y_batch = y[batch_idx]
            if augment_rare:
                for i in range(len(X_batch)):
                    class_label = np.argmax(y_batch[i])
                    if class_label in [1, 2, 3] and np.random.rand() < 0.9:
                        X_batch[i] = augment_eeg(X_batch[i])
            yield X_batch, y_batch

def data_generator_eval(X, y, groups, batch_size=64):
    idx = np.arange(len(X))
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx], groups[batch_idx]

def get_label_alzheimer(group, mmse):
    """Étiquettes spécifiques pour Alzheimer uniquement."""
    if group != "A":  # Ne garder que les patients Alzheimer
        return -1
    if mmse >= 24:
        return 0  # Sain ou très léger
    elif 19 <= mmse <= 23:
        return 1  # Début
    elif 10 <= mmse <= 18:
        return 2  # Modéré
    elif mmse < 10:
        return 3  # Avancé
    return -1

def faiss_deduplicate_and_balance(X, y, patient_ids, max_samples=10_000_000, target_per_class=20000, 
                                  faiss_file="faiss_index.bin", metadata_file="faiss_metadata.pkl", 
                                  use_gpu=False, gpu_resources=None):
    X_flat = X.reshape(len(X), -1).astype(np.float32)
    dim = X_flat.shape[1]
    index = create_or_load_faiss_index(dim, faiss_file, use_gpu, gpu_resources)
    
    if os.path.exists(metadata_file):
        X_existing, y_existing, pids_existing = joblib.load(metadata_file)
        print(f"✅ Métadonnées FAISS chargées depuis {metadata_file} ({len(X_existing)} éléments)")
    else:
        X_existing, y_existing, pids_existing = np.array([]), np.array([]), np.array([])

    index.add(X_flat)
    X_combined = np.concatenate([X_existing, X]) if X_existing.size else X
    y_combined = np.concatenate([y_existing, y]) if y_existing.size else y
    pids_combined = np.concatenate([pids_existing, patient_ids]) if pids_existing.size else patient_ids

    k = 5
    distances, indices = index.search(X_flat, k)
    
    seuil_doublon = 0.01
    keep_mask = np.ones(len(X_combined), dtype=bool)
    for i in range(len(X_flat)):
        new_idx = len(X_existing) + i
        if not keep_mask[new_idx]:
            continue
        neighbors = indices[i, 1:]
        neighbor_dists = distances[i, 1:]
        for j, dist in zip(neighbors, neighbor_dists):
            if j < len(X_combined) and dist < seuil_doublon and keep_mask[j]:
                keep_mask[j] = False

    X_dedup = X_combined[keep_mask]
    y_dedup = y_combined[keep_mask]
    pids_dedup = pids_combined[keep_mask]
    print(f"✅ Après déduplication FAISS : {len(X_dedup)} segments restants")

    final_indices = []
    for class_label in range(4):
        class_indices = np.where(y_dedup == class_label)[0]
        if len(class_indices) > target_per_class:
            selected_indices = np.random.choice(class_indices, target_per_class, replace=False)
        else:
            selected_indices = class_indices
        final_indices.extend(selected_indices)

    X_final = X_dedup[final_indices]
    y_final = y_dedup[final_indices]
    pids_final = pids_dedup[final_indices]

    if len(X_final) > max_samples:
        final_indices = np.random.choice(len(X_final), max_samples, replace=False)
        X_final = X_final[final_indices]
        y_final = y_final[final_indices]
        pids_final = pids_final[final_indices]

    print(f"✅ Après équilibrage FAISS : {len(X_final)} segments au total")
    if use_gpu and gpu_resources is not None:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, faiss_file)
    print(f"✅ Index FAISS sauvegardé : {faiss_file}")
    joblib.dump((X_final, y_final, pids_final), metadata_file)
    print(f"✅ Métadonnées FAISS sauvegardées : {metadata_file}")
    return X_final, y_final, pids_final

# =============================================================================
#                        >>>  Paramètres & Données  <<<
# =============================================================================

device, gpu_resources = setup_device_and_faiss()
fs = 500
num_electrodes = 19
samples = 768
max_samples = 10_000_000

data_dir = "/workspace/memory_os_ai/alz/"
participants_file = os.path.join(data_dir, "participants.tsv")
faiss_file = os.path.join(data_dir, "faiss_index.bin")
metadata_file = os.path.join(data_dir, "faiss_metadata.pkl")
participants = pd.read_csv(participants_file, sep="\t")

X_real, y_real, patient_ids = [], [], []
flat_channels_counter = np.zeros(num_electrodes)

for sub_dir in os.listdir(data_dir):
    if sub_dir.startswith("sub-"):
        eeg_path = os.path.join(data_dir, sub_dir, "eeg", f"{sub_dir}_task-eyesclosed_eeg.set")
        if os.path.exists(eeg_path):
            try:
                participant = participants[participants["participant_id"] == sub_dir].iloc[0]
                if participant["Group"] != "A":  # Filtrer pour ne garder que "A" (Alzheimer)
                    print(f"ℹ️ Ignoré {sub_dir} (Groupe {participant['Group']} ≠ A)")
                    continue

                raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
                raw.resample(fs)
                data_eeg = raw.get_data(picks=raw.ch_names[:num_electrodes])
                data_eeg, flat_mask = sanitize_eeg(data_eeg, verbose=True)
                flat_channels_counter += flat_mask.astype(int)
                total_samples = data_eeg.shape[1]
                num_segments = total_samples // samples

                patient_id = sub_dir
                label_num = get_label_alzheimer(participant["Group"], participant["MMSE"])
                if label_num == -1:
                    print(f"⚠️ Label invalide pour {sub_dir}, ignoré")
                    continue

                for i in range(num_segments):
                    start = i * samples
                    end = start + samples
                    segment = data_eeg[:, start:end]
                    if np.all(np.std(segment, axis=1) < 1e-6):
                        print(f"⚠️ Segment trop plat dans {sub_dir} (segment {i}), ignoré")
                        continue
                    filtered = ukf_physical_adaptive(segment, fs=fs)
                    filtered_segment = filtered.T
                    normalized_segment = normalize_eeg(filtered_segment)
                    X_real.append(normalized_segment)
                    y_real.append(label_num)
                    patient_ids.append(patient_id)
            except Exception as e:
                print(f"⚠️ Erreur sur {eeg_path}: {e}")
                continue

X_real = np.array(X_real)
y_real = np.array(y_real)
patient_ids = np.array(patient_ids)
print(f"✅ Chargé {len(X_real)} segments réels Alzheimer (de {len(np.unique(patient_ids))} sujets)")

# Visualisation des canaux plats
plt.figure(figsize=(10, 6))
sns.barplot(x=range(num_electrodes), y=flat_channels_counter)
plt.title("Électrodes les plus souvent plates (Alzheimer uniquement)")
plt.xlabel("Numéro de l'électrode")
plt.ylabel("Nombre de détections comme plat")
plt.savefig(os.path.join(data_dir, "flat_channel_barplot_alzheimer.png"))
print(f"✅ Barplot des canaux plats sauvegardé sous {os.path.join(data_dir, 'flat_channel_barplot_alzheimer.png')}")

data_file = os.path.join(data_dir, "eeg_data_alzheimer.pkl")
model_file = os.path.join(data_dir, "alz_model_alzheimer.keras")

if os.path.exists(data_file):
    X_loaded, y_loaded, pids_loaded = joblib.load(data_file)
    X = np.concatenate([X_loaded, X_real])
    y = np.concatenate([y_loaded, y_real])
    patient_ids = np.concatenate([pids_loaded, patient_ids])
    print(f"✅ Chargement incrémental de {len(X_loaded)} segments")
else:
    X = X_real
    y = y_real
    print("✅ Aucun fichier existant, on part des données réelles Alzheimer")

X_sim, y_sim, pids_sim = [], [], []
alpha_sain, beta_sain = 80, 20
p_sain = np.random.beta(alpha_sain, beta_sain)
num_sain = int(len(X_real) * p_sain) if len(X_real) > 0 else 10000
class_target_counts = {"Début": 20000, "Modéré": 20000, "Avancé": 20000}
print(f"✅ Proportion 'Sain' : {p_sain*100:.2f}% -> {num_sain} segments simulés Sains")

states = {"Sain": 0.8, "Début": 1.5, "Modéré": 2.0, "Avancé": 2.5}
std_dev = 0.3
ar_params = [1, -0.5]
ma_params = [1]

for i in range(num_sain):
    eeg_data = np.zeros((num_electrodes, samples))
    for ch in range(num_electrodes):
        raw_signal = generate_arima_eeg(states["Sain"], std_dev, ar_params, ma_params)
        noisy_signal = raw_signal + np.random.normal(0, 0.01, samples)
        eeg_data[ch, :] = noisy_signal
        if np.random.rand() < 0.3:
            artifact_duration = int(0.2 * fs)
            start_art = np.random.randint(0, samples - artifact_duration)
            eeg_data[ch, start_art:start_art+artifact_duration] += np.sin(
                2 * np.pi * 50 * np.linspace(0, 0.2, artifact_duration)
            )
    filtered = ukf_physical_adaptive(eeg_data, fs=fs)
    normed_segment = normalize_eeg(filtered.T)
    X_sim.append(normed_segment)
    y_sim.append(0)
    pids_sim.append(f"sim-sain-{i:06d}")

for progression, count in class_target_counts.items():
    for i in range(count):
        eeg_data = np.zeros((num_electrodes, samples))
        for ch in range(num_electrodes):
            raw_signal = generate_arima_eeg(states[progression], std_dev, ar_params, ma_params)
            noisy_signal = raw_signal + np.random.normal(0, 0.01, samples)
            eeg_data[ch, :] = noisy_signal
            if np.random.rand() < 0.3:
                artifact_duration = int(0.2 * fs)
                start_art = np.random.randint(0, samples - artifact_duration)
                eeg_data[ch, start_art:start_art+artifact_duration] += np.sin(
                    2*np.pi*50*np.linspace(0, 0.2, artifact_duration)
                )
        filtered = ukf_physical_adaptive(eeg_data, fs=fs)
        normed_segment = normalize_eeg(filtered.T)
        X_sim.append(normed_segment)
        y_sim.append(["Sain", "Début", "Modéré", "Avancé"].index(progression))
        pids_sim.append(f"sim-{progression.lower()}-{i:06d}")

X = np.concatenate([X, np.array(X_sim)])
y = np.concatenate([y, np.array(y_sim)])
patient_ids = np.concatenate([patient_ids, np.array(pids_sim)])

X, y, patient_ids = faiss_deduplicate_and_balance(
    X, y, patient_ids, max_samples=max_samples, target_per_class=20000, 
    faiss_file=faiss_file, metadata_file=metadata_file, 
    use_gpu=(device == "GPU"), gpu_resources=gpu_resources
)
joblib.dump((X, y, patient_ids), data_file)
print(f"✅ Données incrémentales sauvegardées : {data_file}")

# =============================================================================
#                           >>>  Modèle & Entraînement  <<<
# =============================================================================

num_classes = 4
y_onehot = to_categorical(y, num_classes=num_classes)

def attention_block(inputs, name_suffix=""):
    avg_pool = GlobalAveragePooling1D()(inputs)
    att = Dense(inputs.shape[1], activation='softmax', name=f"attention_dense_{name_suffix}")(avg_pool)
    att = Reshape((inputs.shape[1], 1))(att)
    out = Multiply()([inputs, att])
    return out

input_layer = Input(shape=(samples, num_electrodes))
x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = BatchNormalization()(x)
x = attention_block(x, name_suffix="alz")
x = Bidirectional(LSTM(64))(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

clf = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.001)
clf.compile(optimizer=optimizer, loss=CategoricalFocalCrossentropy(gamma=6), metrics=['accuracy'])

if os.path.exists(model_file):
    try:
        old_model = load_model(model_file, compile=False)
        print(f"✅ Modèle chargé depuis {model_file}")
        clf = old_model
        clf.compile(optimizer=optimizer, loss=CategoricalFocalCrossentropy(gamma=6), metrics=['accuracy'])
    except Exception as e:
        print(f"⚠️ Erreur de chargement : {e}. Nouveau modèle initialisé.")
else:
    print("✅ Nouveau modèle initialisé")

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))

X_train_raw, X_test_raw = X[train_idx], X[test_idx]
y_train_raw, y_test_raw = y_onehot[train_idx], y_onehot[test_idx]
pids_train, pids_test = patient_ids[train_idx], patient_ids[test_idx]

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx2, val_idx = next(gss_val.split(X_train_raw, y_train_raw, groups=pids_train))

X_train, X_val = X_train_raw[train_idx2], X_train_raw[val_idx]
y_train, y_val = y_train_raw[train_idx2], y_train_raw[val_idx]
pids_train_split, pids_val = pids_train[train_idx2], pids_train[val_idx]

print(f"✅ Taille : Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test_raw)}")

batch_size = 64
train_gen = data_generator(X_train, y_train, pids_train_split, batch_size=batch_size, augment_rare=True)
steps_per_epoch = len(X_train) // batch_size

cw = compute_class_weight(class_weight='balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weights = dict(enumerate(cw))
print(f"✅ Poids de classe : {class_weights}")

checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

epochs = 30
history = clf.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

clf = load_model(model_file, compile=False)
clf.compile(optimizer=optimizer, loss=CategoricalFocalCrossentropy(gamma=6), metrics=['accuracy'])

# =============================================================================
#                     >>>  Évaluation & Visualisations  <<<
# =============================================================================

test_gen = data_generator_eval(X_test_raw, y_test_raw, pids_test, batch_size=batch_size)
y_pred_all, y_test_all, pids_test_all = [], [], []
num_batches = int(np.ceil(len(X_test_raw)/batch_size))
for _ in range(num_batches):
    X_batch, y_batch, pids_batch = next(test_gen)
    preds = clf.predict(X_batch, verbose=0)
    y_pred_all.append(preds)
    y_test_all.append(y_batch)
    pids_test_all.append(pids_batch)

y_pred = np.concatenate(y_pred_all)
y_test_onehot = np.concatenate(y_test_all)
pids_test_array = np.concatenate(pids_test_all)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_onehot, axis=1)

uncertain_cases = np.max(y_pred, axis=1) < 0.5
print(f"✅ Cas incertains détectés : {np.sum(uncertain_cases)} segments (proba max < 0.5)")

patient_preds = defaultdict(list)
for pid, pred_class in zip(pids_test_array, y_pred_classes):
    patient_preds[pid].append(pred_class)

patient_results = {pid: Counter(preds).most_common(1)[0][0] for pid, preds in patient_preds.items()}

def get_patient_label(pid):
    if pid.startswith("sub-"):
        row = participants[participants["participant_id"] == pid].iloc[0]
        return get_label_alzheimer(row["Group"], row["MMSE"])
    else:
        if "sain" in pid:
            return 0
        elif "début" in pid:
            return 1
        elif "modéré" in pid:
            return 2
        elif "avancé" in pid:
            return 3
        return 0

y_test_patient = [get_patient_label(pid) for pid in patient_results.keys()]
y_pred_patient = list(patient_results.values())

accuracy_patient = accuracy_score(y_test_patient, y_pred_patient)
conf_matrix_patient = confusion_matrix(y_test_patient, y_pred_patient)
class_report_patient = classification_report(y_test_patient, y_pred_patient)
f1_macro_patient = f1_score(y_test_patient, y_pred_patient, average='macro')

print(f"\n✅ Précision finale (par patient, Alzheimer uniquement) : {accuracy_patient * 100:.2f}%")
print("📊 Matrice de confusion (par patient) :\n", conf_matrix_patient)
print("\n📊 Rapport de classification (par patient) :\n", class_report_patient)
print(f"✅ Score F1 macro (par patient) : {f1_macro_patient * 100:.2f}%")

alz_truth = [1 if t in [1, 2, 3] else 0 for t in y_test_patient]
alz_pred = [1 if p in [1, 2, 3] else 0 for p in y_pred_patient]
alz_recall = recall_score(alz_truth, alz_pred)
print(f"🧠 Sensibilité Alzheimer (global) : {alz_recall * 100:.2f}%")

stimulation_recommendations = {
    0: "Aucune stimulation nécessaire.",
    1: "10 min de VR Alpha pour synchronisation.",
    2: "15 min de VR Theta + sommeil optimisé.",
    3: "20 min de VR Delta + consultation urgente."
}
class_names = ["Sain", "Début Alzheimer", "Modéré Alzheimer", "Avancé Alzheimer"]
for pred in set(y_pred_patient):
    print(f"🔹 Pour {class_names[pred]} : {stimulation_recommendations[pred]}")

plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix_patient, cmap="Blues")
plt.title("Matrice de confusion Alz (par patient, Alzheimer uniquement)")
plt.colorbar()
plt.xlabel("Prédictions")
plt.ylabel("Vraies classes")
plt.xticks(ticks=[0, 1, 2, 3], labels=["Sain", "Début", "Modéré", "Avancé"])
plt.yticks(ticks=[0, 1, 2, 3], labels=["Sain", "Début", "Modéré", "Avancé"])
plt.savefig(os.path.join(data_dir, "alz_confusion_matrix_patient_alzheimer.png"))
print(f"✅ Matrice de confusion sauvegardée sous {os.path.join(data_dir, 'alz_confusion_matrix_patient_alzheimer.png')}")

plt.figure(figsize=(10, 6))
max_probs = np.max(y_pred, axis=1)
plt.hist(max_probs, bins=20, range=(0, 1), color='skyblue', edgecolor='black')
plt.title("Distribution des probabilités maximales prédites (Alzheimer uniquement)")
plt.xlabel("Probabilité maximale")
plt.ylabel("Nombre de prédictions")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(data_dir, "probability_histogram_alzheimer.png"))
print(f"✅ Histogramme des probabilités sauvegardé sous {os.path.join(data_dir, 'probability_histogram_alzheimer.png')}")

onnx_file = os.path.join(data_dir, "alz_model_alzheimer.onnx")
model_proto, _ = tf2onnx.convert.from_keras(clf, output_path=onnx_file)
print(f"✅ Modèle exporté en ONNX sous {onnx_file}")

tflite_file = os.path.join(data_dir, "alz_model_alzheimer.tflite")
converter = tf.lite.TFLiteConverter.from_keras_model(clf)
tflite_model = converter.convert()
with open(tflite_file, 'wb') as f:
    f.write(tflite_model)
print(f"✅ Modèle exporté en TFLite sous {tflite_file}")


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
