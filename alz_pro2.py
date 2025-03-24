# alz_pro2.py – Modèle IA simplifié de détection Alzheimer via EEG simulé
# Auteur : Kocupyr Romain
# Licence : Creative Commons BY-NC-SA 4.0
# https://creativecommons.org/licenses/by-nc-sa/4.0/

import numpy as np
import mne
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import faiss
import multiprocessing
import joblib
import tf2onnx
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
from statsmodels.tsa.arima_process import ArmaProcess
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

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
    """Création ou chargement de l'index FAISS (ou sur GPU si possible)."""
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

def add_to_faiss_index_in_batches(index, data, batch_size=100_000):
    """Ajout des vecteurs à FAISS par batch pour éviter l’excès de mémoire."""
    n = len(data)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        index.add(data[start:end])

def search_faiss_index_in_batches(index, data, k=5, batch_size=100_000):
    """Recherche FAISS par batch pour optimiser mémoire & paralléliser si GPU."""
    distances_all = []
    indices_all = []
    n = len(data)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        d, i = index.search(data[start:end], k)
        distances_all.append(d)
        indices_all.append(i)
    return np.concatenate(distances_all, axis=0), np.concatenate(indices_all, axis=0)

def sanitize_eeg(eeg_data, verbose=False):
    """Nettoie les NaN/infs et gère les segments plats."""
    eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
    eeg_data = np.clip(eeg_data, -1e6, 1e6)
    stds = np.std(eeg_data, axis=1)
    too_flat = stds < 1e-10
    if np.any(too_flat) and verbose:
        print(f"⚠️ Segments plats détectés dans {np.where(too_flat)[0]}, ajout de bruit léger")
        for i in np.where(too_flat)[0]:
            eeg_data[i] += np.random.normal(0, 1e-6, eeg_data.shape[1])
    return eeg_data, too_flat

def compute_spectral_features(data, fs, window_size):
    """Extraction des fréquences dominantes, band-powers et stabilité (via stft scipy)."""
    data, _ = sanitize_eeg(data, verbose=False)
    if data.shape[1] < window_size:
        freqs, _, Zxx = stft(data, fs=fs, nperseg=data.shape[1], noverlap=data.shape[1]//2)
    else:
        freqs, _, Zxx = stft(data, fs=fs, nperseg=window_size, noverlap=window_size//2)
    
    power = np.mean(np.abs(Zxx)**2, axis=2)
    power, _ = sanitize_eeg(power, verbose=False)
    dominant_freqs = freqs[np.argmax(power, axis=1)]
    
    bands = {'delta': (0, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}
    band_powers = np.zeros((data.shape[0], len(bands)))
    for i, (band, (fmin, fmax)) in enumerate(bands.items()):
        idx = (freqs >= fmin) & (freqs <= fmax)
        band_powers[:, i] = np.mean(power[:, idx], axis=1)
    band_powers, _ = sanitize_eeg(band_powers, verbose=False)
    
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
    """Corrélation moyenne inter-canaux."""
    data, _ = sanitize_eeg(data, verbose=False)
    if data.shape[1] < 2:
        return 0.1
    corr_matrix = np.corrcoef(data)
    if np.any(np.isnan(corr_matrix)) or np.any(np.isinf(corr_matrix)):
        return 0.1
    off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    return np.mean(np.abs(off_diag))

# ---------------------------------------------------------------------------
# >>> 1) Fonctions physiques en TensorFlow (fx, hx)  <<<
# ---------------------------------------------------------------------------
@tf.function(reduce_retracing=True)
def physical_eeg_model_tf(x, dt, freqs, coupling, band_powers):
    """
    Réplique en TF le physical_eeg_model() (équations simples).
    x.shape=(n_channels,)
    """
    n_channels = tf.shape(x)[0]
    # dx_vals[i] = self_term + coupling_term
    # Vectorisation au lieu de boucles python
    mean_x = tf.reduce_mean(x)  # pour calculer (mean_x - x[i]) sur chaque canal

    # On calcule alpha_power canal par canal
    sum_bands = tf.reduce_sum(band_powers, axis=1) + 1e-8  # shape=(n_channels,)
    alpha_pwr = band_powers[:, 2] / sum_bands  # alpha band

    # amplitude_mod[i] = clip(1 + alpha_pwr[i], 0.1, 10)
    amplitude_mod = tf.clip_by_value(1.0 + alpha_pwr, 0.1, 10.0)

    # self_term[i] = -omega^2 * sin(omega * dt) * x[i] * amplitude_mod[i]
    # coupling_term[i] = coupling * (mean_x - x[i])
    # => dx[i] = self_term[i] + coupling_term[i]
    omega = 2.0 * np.pi * freqs
    sin_part = tf.math.sin(omega * dt)  # shape=(n_channels,)

    self_term = - (omega**2) * sin_part * x * amplitude_mod
    coupling_term = coupling * (mean_x - x)
    dx_vals = self_term + coupling_term

    return x + dt * dx_vals

@tf.function(reduce_retracing=True)
def measurement_function_tf(x):
    """hx = identité en TF"""
    return x

# ---------------------------------------------------------------------------
# >>> 2) Fonctions Merwe (sigma points) et unscented transform  <<<
# ---------------------------------------------------------------------------
@tf.function(reduce_retracing=True)
def merwe_params_tf(n, alpha=0.3, beta=2.0, kappa=1.0):
    """
    Calcule lambda, Wm, Wc pour le MerweScaledSigmaPoints en mode TensorFlow.
    """
    n_float = tf.cast(n, tf.float32)
    alpha_tf = tf.constant(alpha, dtype=tf.float32)
    beta_tf = tf.constant(beta, dtype=tf.float32)
    kappa_tf = tf.constant(kappa, dtype=tf.float32)

    lambda_ = alpha_tf**2 * (n_float + kappa_tf) - n_float
    c = n_float + lambda_

    # Wm / Wc shape = (2n+1,)
    Wm = tf.fill((2*n+1,), 1.0 / (2.0 * c))
    Wc = tf.fill((2*n+1,), 1.0 / (2.0 * c))

    # Wm[0] = lambda_/c
    Wm_0 = lambda_/c
    # Wc[0] = lambda_/c + (1 - alpha^2 + beta)
    Wc_0 = lambda_/c + (1.0 - alpha_tf**2 + beta_tf)

    Wm = tf.tensor_scatter_nd_update(Wm, [[0]], [Wm_0])
    Wc = tf.tensor_scatter_nd_update(Wc, [[0]], [Wc_0])
    return lambda_, Wm, Wc

@tf.function(reduce_retracing=True)
def generate_sigma_points_tf(x, P, alpha=0.3, beta=2.0, kappa=1.0):
    """
    Génération vectorisée des sigma points (Merwe) en TensorFlow, 
    x.shape = (n,), P.shape = (n,n).
    Retourne un tenseur (2n+1, n).
    """
    n = tf.shape(x)[0]
    lambda_, Wm, Wc = merwe_params_tf(n, alpha, beta, kappa)
    c = tf.cast(n, tf.float32) + lambda_

    # Décomposition de P * c
    L = tf.linalg.cholesky(P * c)  # (n, n)

    # On veut x.shape => (1,n) pour broadcast
    x_expanded = tf.expand_dims(x, axis=0)  # (1,n)

    # plus  => x + L^T
    # minus => x - L^T
    # => shape (n, n) chacun
    L_t = tf.transpose(L)  # (n,n)
    plus = x_expanded + L_t  # (n,n)
    minus = x_expanded - L_t  # (n,n)

    # Concaténer dans l'ordre :
    sp0 = x_expanded
    sigma_points = tf.concat([sp0, plus, minus], axis=0)  # (2n+1, n)
    return sigma_points

@tf.function(reduce_retracing=True)
def unscented_transform_tf(Xsigma, Wm, Wc, noise_cov=None):
    """
    Applique la transformée unscented : calcule la moyenne et la covariance
    à partir des sigma points propagés.
    - Xsigma : (2n+1, n)
    - Wm, Wc : (2n+1,) weights
    - noise_cov : (n,n) optionnel
    Retourne : mean, cov
    """
    x_mean = tf.reduce_sum(Xsigma * tf.reshape(Wm, (-1,1)), axis=0)  # (n,)
    diff = Xsigma - x_mean  # (2n+1, n)
    diff_expanded = tf.expand_dims(diff, axis=2)  # (2n+1, n, 1)
    wc_reshaped = tf.reshape(Wc, (-1,1,1))        # (2n+1, 1, 1)

    cov_terms = wc_reshaped * tf.matmul(diff_expanded, diff_expanded, transpose_b=True)  # (2n+1, n, n)
    P = tf.reduce_sum(cov_terms, axis=0)  # (n,n)

    if noise_cov is not None:
        P = P + noise_cov
    return x_mean, P

@tf.function(reduce_retracing=True)
def predict_update_ukf_tf(
    x, P, z, Q, R, dt,
    freqs, coupling, band_powers
):
    """
    Fonction unique pour la prédiction ET la mise à jour, 
    évite d'appeler 2 fonctions décorées distinctes dans la boucle.
    - x, P : état et covariance
    - z : mesure
    - Q, R : bruit de process et mesure
    - dt : échantillonnage (tf.float32)
    - freqs, coupling, band_powers : Tenseurs pour la dynamique EEG
    Retourne : x_upd, P_upd
    """

    # ---- 1) PREDICTION ----
    n = tf.shape(x)[0]
    alpha=0.3
    beta=2.0
    kappa=1.0

    # Sigma points
    Xsigma = generate_sigma_points_tf(x, P, alpha, beta, kappa)  # (2n+1, n)
    # On applique fx
    def fx(sp):
        return physical_eeg_model_tf(sp, dt, freqs, coupling, band_powers)
    Xsigma_pred = tf.map_fn(fx, Xsigma)
    # Moy/cov
    _, Wm, Wc = merwe_params_tf(n, alpha, beta, kappa)
    x_pred, P_pred = unscented_transform_tf(Xsigma_pred, Wm, Wc, noise_cov=Q)

    # ---- 2) UPDATE ----
    # On projette Xsigma_pred dans l'espace de mesure hx
    Zsigma_pred = tf.map_fn(measurement_function_tf, Xsigma_pred)
    z_mean, Pz = unscented_transform_tf(Zsigma_pred, Wm, Wc, noise_cov=R)

    diff_x = Xsigma_pred - x_pred
    diff_z = Zsigma_pred - z_mean
    wc_reshaped = tf.reshape(Wc, (-1,1,1))  # (2n+1,1,1)

    diff_x_exp = tf.expand_dims(diff_x, axis=2)   # (2n+1, n, 1)
    diff_z_exp = tf.expand_dims(diff_z, axis=1)   # (2n+1, 1, n)
    Pxz_terms = wc_reshaped * tf.matmul(diff_x_exp, diff_z_exp)
    Pxz = tf.reduce_sum(Pxz_terms, axis=0)  # (n,n)

    Pz_inv = tf.linalg.inv(Pz)
    K = tf.matmul(Pxz, Pz_inv)  # (n,n)

    y = z - z_mean
    x_new = x_pred + tf.matmul(K, tf.expand_dims(y, axis=1))[:, 0]
    P_new = P_pred - tf.matmul(K, tf.matmul(Pz, K, transpose_b=True))

    return x_new, P_new

# ---------------------------------------------------------------------------
# >>> 3) UKF complet adaptatif (dans un loop Python), 1 seul appel TF / itération  <<<
# ---------------------------------------------------------------------------
def ukf_physical_adaptive_tf(eeg_data, fs=500, process_noise_init=0.1, measurement_noise_init=0.1):
    """
    Variante 100% TensorFlow du UKF adaptatif, exploitant le GPU pour la prédiction/mise à jour.
    La partie adaptative (Q, R dynamiques, etc.) reste en Python, 
    on appelle 'predict_update_ukf_tf()' 1 fois par itération => 1 graph.
    """
    eeg_data, _ = sanitize_eeg(eeg_data, verbose=False)
    n_channels, n_samples = eeg_data.shape

    # On définit dt_tf (constant) pour éviter le retracing à chaque itération
    dt_tf = tf.constant(1.0 / fs, dtype=tf.float32)

    # Initialisation x, P
    x_tf = tf.constant(eeg_data[:, 0], dtype=tf.float32)
    P_tf = tf.eye(n_channels, dtype=tf.float32) * process_noise_init

    # Q_base, R_base init
    window_init = eeg_data[:, :min(100, n_samples)]
    if window_init.shape[1] > 1:
        Q_base_np = np.cov(window_init)
        if np.any(np.isnan(Q_base_np)) or np.any(np.isinf(Q_base_np)):
            Q_base_np = np.eye(n_channels) * process_noise_init
    else:
        Q_base_np = np.eye(n_channels) * process_noise_init

    R_base_np = np.eye(n_channels) * measurement_noise_init

    Q_base_tf = tf.constant(Q_base_np, dtype=tf.float32)
    R_base_tf = tf.constant(R_base_np, dtype=tf.float32)

    out = np.zeros_like(eeg_data)
    out[:, 0] = eeg_data[:, 0]

    # Paramètres adaptatifs
    min_window, max_window = 50, 200
    min_coupling, max_coupling = 0.05, 0.2
    window_size = 100
    coupling_window = 0.1
    freqs_window = np.ones(n_channels, dtype=np.float32) * 10.0
    band_powers_window = np.ones((n_channels, 4), dtype=np.float32)

    # On cast pour TF
    coupling_tf = tf.constant(coupling_window, dtype=tf.float32)
    freqs_window_tf = tf.constant(freqs_window, dtype=tf.float32)
    band_powers_window_tf = tf.constant(band_powers_window, dtype=tf.float32)

    for t in range(1, n_samples):
        z_np = eeg_data[:, t]  # mesure
        z_tf = tf.constant(z_np, dtype=tf.float32)

        window_start = max(0, t - window_size)
        window_data = eeg_data[:, window_start:t]

        if window_data.shape[1] >= 10:
            stability, freqs_np, band_powers_np = compute_spectral_features(window_data, fs, window_size)
            window_size = int(min_window + (max_window - min_window) * stability)
            window_size = max(min_window, min(window_size, t))

            correlation = compute_inter_channel_correlation(window_data)
            coupling_window = min_coupling + (max_coupling - min_coupling) * correlation

            cov_np = np.cov(window_data)
            if np.any(np.isnan(cov_np)) or np.any(np.isinf(cov_np)):
                cov_np = np.eye(n_channels) * process_noise_init
            Q_tf = tf.constant(cov_np * process_noise_init, dtype=tf.float32)

            z_diff_np = np.abs(z_np - x_tf.numpy())
            std_window = np.std(window_data, axis=1)
            std_window = np.where(std_window < 1e-8, 1e-8, std_window)
            artifact_factor = np.where(z_diff_np > 3 * std_window, 10.0, 1.0)
            R_adapt_np = R_base_np * artifact_factor[:, np.newaxis]
            R_tf = tf.constant(R_adapt_np, dtype=tf.float32)

            coupling_tf = tf.constant(coupling_window, dtype=tf.float32)
            freqs_window_tf = tf.constant(freqs_np, dtype=tf.float32)
            band_powers_window_tf = tf.constant(band_powers_np, dtype=tf.float32)
        else:
            # Moins de 10 points => Q,R basiques
            Q_tf = Q_base_tf
            R_tf = R_base_tf

        # Appel unique TF => 1 trace (ou peu)
        x_upd, P_upd = predict_update_ukf_tf(
            x_tf, P_tf, z_tf, Q_tf, R_tf, dt_tf,
            freqs_window_tf, coupling_tf, band_powers_window_tf
        )

        x_upd_np = x_upd.numpy()
        if np.any(np.isnan(x_upd_np)) or np.any(np.isinf(x_upd_np)):
            print(f"⚠️ NaN/Inf dans x au timestep {t}, réinitialisation")
            x_upd_np = np.nan_to_num(x_upd_np, nan=0.0, posinf=0.0, neginf=0.0)

        out[:, t] = x_upd_np
        x_tf = tf.constant(x_upd_np, dtype=tf.float32)
        P_tf = P_upd

    return out

def normalize_eeg(segment):
    """
    Normalisation du segment EEG :
    - Remplacement des nan/inf par 0
    - Vérification si segment vide
    - Soustraction de la moyenne
    - Division par l'écart-type (avec std min = 1e-8)
    - Retour final sans nan/inf
    """
    segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    if segment.size == 0:
        return np.zeros_like(segment)
    mean = np.mean(segment, axis=0)
    std = np.std(segment, axis=0)
    std = np.where(std == 0, 1e-8, std)
    normalized = (segment - mean) / std
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

def augment_eeg(segment):
    """Augmentations légères (shift + noise + scale)."""
    shift = np.random.randint(-5, 5)
    segment = np.roll(segment, shift, axis=0)
    noise = np.random.normal(0, 0.01, segment.shape)
    segment += noise
    scale = np.random.uniform(0.98, 1.02)
    segment *= scale
    return segment

def generate_arima_eeg(mean, std, ar, ma, samples=768):
    """Génération d'une suite de type ARIMA pour simuler un canal EEG."""
    arima_process = ArmaProcess(np.array(ar), np.array(ma)).generate_sample(nsample=samples)
    return mean + std * arima_process

def data_generator(X, y, groups, batch_size=64, augment_rare=True):
    """Générateur de batches pour l'entraînement."""
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
    """Générateur pour l'évaluation (pas d'augmentation)."""
    idx = np.arange(len(X))
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx], groups[batch_idx]

def get_label_alzheimer(group, mmse):
    """
    Renvoie une étiquette pour un patient Alzheimer
    Retourne -1 si le patient n'est pas A/AD ou label impossible.
    """
    if group not in ["A", "AD"]:
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
    """Déduplication + équilibrage par FAISS (avec batch)."""
    X_flat = X.reshape(len(X), -1).astype(np.float32)
    dim = X_flat.shape[1]
    index = create_or_load_faiss_index(dim, faiss_file, use_gpu, gpu_resources)
    
    if os.path.exists(metadata_file):
        X_existing, y_existing, pids_existing = joblib.load(metadata_file)
        print(f"✅ Métadonnées FAISS chargées depuis {metadata_file} ({len(X_existing)} éléments)")
    else:
        X_existing, y_existing, pids_existing = np.array([]), np.array([]), np.array([])

    add_to_faiss_index_in_batches(index, X_flat)
    X_combined = np.concatenate([X_existing, X]) if X_existing.size else X
    y_combined = np.concatenate([y_existing, y]) if y_existing.size else y
    pids_combined = np.concatenate([pids_existing, patient_ids]) if pids_existing.size else patient_ids

    k = 5
    distances, indices = search_faiss_index_in_batches(index, X_flat, k=k)

    seuil_doublon = 0.01
    keep_mask = np.ones(len(X_combined), dtype=bool)
    offset = len(X_existing)

    for i in range(len(X_flat)):
        new_idx = offset + i
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

# >>> Traitement patient par patient (A ou AD), parallélisation sur les segments <<<
for sub_dir in os.listdir(data_dir):
    if sub_dir.startswith("sub-"):
        eeg_path = os.path.join(data_dir, sub_dir, "eeg", f"{sub_dir}_task-eyesclosed_eeg.set")
        if os.path.exists(eeg_path):
            try:
                participant = participants[participants["participant_id"] == sub_dir].iloc[0]
                if participant["Group"] not in ["A", "AD"]:
                    print(f"ℹ️ Ignoré {sub_dir} (Groupe {participant['Group']} ≠ A/AD)")
                    continue

                raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
                raw.resample(fs)  # "Sampling frequency of the instance is already 500.0..."
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

                segments_data = []
                for i in range(num_segments):
                    start = i * samples
                    end = start + samples
                    segment = data_eeg[:, start:end]
                    if np.all(np.std(segment, axis=1) < 1e-6):
                        print(f"⚠️ Segment trop plat dans {sub_dir} (segment {i}), ignoré")
                        continue
                    segments_data.append(segment)
                
                def process_segment(seg):
                    return ukf_physical_adaptive_tf(seg, fs=fs, 
                                                    process_noise_init=0.1,
                                                    measurement_noise_init=0.1)

                with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    results = list(executor.map(process_segment, segments_data))
                
                for filtered in results:
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
print(f"✅ Chargé {len(X_real)} segments réels Alzheimer (A/AD) (de {len(np.unique(patient_ids))} sujets)")

plt.figure(figsize=(10, 6))
sns.barplot(x=list(range(num_electrodes)), y=flat_channels_counter)
plt.title("Électrodes les plus souvent plates (A/AD uniquement)")
plt.xlabel("Numéro de l'électrode")
plt.ylabel("Nombre de détections comme plat")
plt.savefig(os.path.join(data_dir, "flat_channel_barplot_alzheimer.png"))
print(f"✅ Barplot des canaux plats sauvegardé : {os.path.join(data_dir, 'flat_channel_barplot_alzheimer.png')}")

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

# =============================================================================
#               >>>  Génération de données simulées (4 classes)  <<<
# =============================================================================
X_sim, y_sim, pids_sim = [], [], []

alpha_sain, beta_sain = 80, 20
p_sain = np.random.beta(alpha_sain, beta_sain)
num_sain = int(len(X_real) * p_sain) if len(X_real) > 0 else 10000
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
                2*np.pi*50*np.linspace(0, 0.2, artifact_duration)
            )
    filtered_sim = ukf_physical_adaptive_tf(eeg_data, fs=fs)
    normed_segment = normalize_eeg(filtered_sim.T)
    X_sim.append(normed_segment)
    y_sim.append(0)
    pids_sim.append(f"sim-sain-{i:06d}")

class_target_counts = {"Début": 20000, "Modéré": 20000, "Avancé": 20000}
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
        filtered_sim = ukf_physical_adaptive_tf(eeg_data, fs=fs)
        normed_segment = normalize_eeg(filtered_sim.T)
        y_sim.append(["Sain", "Début", "Modéré", "Avancé"].index(progression))
        X_sim.append(normed_segment)
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

cw = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(np.argmax(y_train, axis=1)), 
    y=np.argmax(y_train, axis=1)
)
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

print(f"\n✅ Précision finale (par patient, A/AD) : {accuracy_patient * 100:.2f}%")
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
plt.title("Matrice de confusion Alz (par patient, A/AD)")
plt.colorbar()
plt.xlabel("Prédictions")
plt.ylabel("Vraies classes")
plt.xticks(ticks=[0, 1, 2, 3], labels=["Sain", "Début", "Modéré", "Avancé"])
plt.yticks(ticks=[0, 1, 2, 3], labels=["Sain", "Début", "Modéré", "Avancé"])
plt.savefig(os.path.join(data_dir, "alz_confusion_matrix_patient_alzheimer.png"))
print(f"✅ Matrice de confusion sauvegardée : {os.path.join(data_dir, 'alz_confusion_matrix_patient_alzheimer.png')}")

plt.figure(figsize=(10, 6))
max_probs = np.max(y_pred, axis=1)
plt.hist(max_probs, bins=20, range=(0, 1), color='skyblue', edgecolor='black')
plt.title("Distribution des probabilités maximales prédites (A/AD)")
plt.xlabel("Probabilité maximale")
plt.ylabel("Nombre de prédictions")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(data_dir, "probability_histogram_alzheimer.png"))
print(f"✅ Histogramme des probabilités sauvegardé : {os.path.join(data_dir, 'probability_histogram_alzheimer.png')}")

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
# Ce script "alz_pro2.py" fait partie du projet Alzheimer EEG AI Assistant,
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
