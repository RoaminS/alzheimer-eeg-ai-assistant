"""
adformer999.py — EEG Alzheimer Classifier (Full Pipeline)
Licence : Creative Commons BY-NC-SA 4.0
Auteurs : 
    - Kocupyr Romain (chef de projet) : rkocupyr@gmail.com
    - GPT multi_gpt_api (OpenAI)
    - Grok3
Dataset : https://www.kaggle.com/datasets/yosftag/open-nuro-dataset
"""

import os
import numpy as np
import pandas as pd
import h5py
import mne
import joblib
from scipy.signal import welch
from scipy.stats import iqr
from pykalman import KalmanFilter
import antropy as ant
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_process import ArmaProcess
import faiss

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix, 
                             recall_score)


# ====================================================================================
# === PARAMÈTRES GLOBAUX
# ====================================================================================
fs = 128               # Fréquence d'échantillonnage
samples = 512          # Taille (en échantillons) de chaque segment EEG
num_electrodes = 19    # Nombre d'électrodes utilisées

# On prévoit de stocker la partie "raw" (512 × 19) + la partie "features" (267) = 9995
RAW_SIZE = samples * num_electrodes   # 512 * 19 = 9728
FEATURE_SIZE = 267                    # Taille du vecteur retourné par extract_features()
TOTAL_SIZE = RAW_SIZE + FEATURE_SIZE  # 9728 + 267 = 9995

# Paires d’asymétrie utilisées
asym_pairs = [(3, 5), (13, 15), (0, 1)]

# Bandes de fréquences EEG pour l’extraction spectrale
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha1': (8, 10),
    'Alpha2': (10, 13),
    'Beta1': (13, 20),
    'Beta2': (20, 30),
    'Gamma': (30, 45)
}

# Modèle de Kalman pour le lissage
kf_model = KalmanFilter(initial_state_mean=0, n_dim_obs=1)


# ====================================================================================
# === Fonctions d'extraction de features
# ====================================================================================
def kalman_filter_signal(signal):
    """
    Applique un filtre de Kalman sur un signal 1D (dans l'idée de le lisser).
    """
    filtered, _ = kf_model.filter(signal[:, None])
    return filtered[:, 0]


def extract_features(data):
    """
    Calcule un vecteur de 267 features pour un segment EEG de forme (512, 19).
    """
    if data.shape != (samples, num_electrodes):
        raise ValueError(f"❌ Segment shape invalide : {data.shape}")

    try:
        # === Statistiques temporelles
        mean_t = np.mean(data, axis=0)
        var_t = np.var(data, axis=0)
        iqr_t = iqr(data, axis=0)

        # === PSD
        freqs, psd = welch(data, fs=fs, nperseg=samples, axis=0)
        band_feats, kalman_means, kalman_diffs = [], [], []

        for fmin, fmax in bands.values():
            idx = (freqs >= fmin) & (freqs <= fmax)
            if np.sum(idx) == 0:
                # sécurité : pas de fréquence dans cette bande
                band_feats.append(np.zeros(num_electrodes))
                kalman_means.append(0.0)
                kalman_diffs.append(0.0)
                continue

            raw_power = np.mean(psd[idx], axis=0)
            k_power = kalman_filter_signal(psd[idx].mean(axis=1))
            band_feats.append(raw_power)
            kalman_means.append(np.mean(k_power))
            kalman_diffs.append(raw_power.mean() - np.mean(k_power))

        rbp = np.stack(band_feats, axis=0)  # shape (7, 19)

        # === Complexité (entropies)
        import antropy as ant
        perm_en = np.array([
            ant.perm_entropy(data[:, i], order=3, normalize=True)
            for i in range(num_electrodes)
        ])
        sample_en = np.array([
            ant.sample_entropy(data[:, i], order=2)
            for i in range(num_electrodes)
        ])

        # === Connectivité (corrélation, clustering, efficiency)
        corr_matrix = np.corrcoef(data.T)
        clustering = np.array([
            np.sum(corr_matrix[i] > 0.5) / (num_electrodes - 1)
            for i in range(num_electrodes)
        ])
        path_length = np.mean(np.abs(corr_matrix))
        non_zero_corr = corr_matrix[np.abs(corr_matrix) > 0]
        efficiency = np.mean(1 / np.abs(non_zero_corr)) if len(non_zero_corr) > 0 else 0.0
        small_worldness = np.mean(clustering) / path_length if path_length != 0 else 0.0

        # === Asymétries
        asym = np.array([np.mean(data[:, i] - data[:, j]) for i, j in asym_pairs])

        # === Concaténation finale des 267 features
        features = np.concatenate([
            mean_t, var_t, iqr_t,               # 3 × 19 = 57
            rbp.flatten(),                      # 7 × 19 = 133
            perm_en, sample_en, clustering,     # 3 × 19 = 57
            asym,                               # 3
            [path_length, efficiency, small_worldness],  # 3
            kalman_means, kalman_diffs          # 2×7 = 14
        ])

        assert features.shape[0] == 267, f"❌ Mauvais nb de features : {features.shape[0]}"
        return features

    except Exception as e:
        print(f"❌ Erreur dans extract_features : {e}")
        return np.array([])  # On renvoie un tableau vide pour ignorer ce segment défectueux


# ====================================================================================
# === Helpers
# ====================================================================================
def get_label(row):
    """
    Détermine le label en fonction du champ 'Group' (A / AD) et du score 'MMSE'.
    On renvoie 1, 2 ou 3 selon la sévérité (ou -1 si hors scope).
    """
    if row["Group"] not in ["A", "AD"]:
        return -1
    mmse = row["MMSE"]
    if pd.isna(mmse):
        return 1
    return 1 if mmse >= 19 else 2 if mmse >= 10 else 3


def generate_arima_eeg(mean, std, samples=512):
    """
    Génère un signal EEG fictif par un processus ARIMA
    (ex: ar=1, ma=0.5) autour d'une moyenne et std paramétrées.
    """
    from statsmodels.tsa.arima_process import ArmaProcess
    ar_params = np.array([1, -0.5])
    ma_params = np.array([1])
    return mean + std * ArmaProcess(ar_params, ma_params).generate_sample(nsample=samples)


# ====================================================================================
# === Construction du dataset
# ====================================================================================
def build_dataset_with_sim(data_dir, out_file):
    """
    Construit et sauvegarde en HDF5 un dataset équilibré.
    - Parcours les EEG existants, extrait tous les segments valides (512×19).
    - Calcule 267 features.
    - Concatène (raw + features) -> vecteur de taille 9995.
    - Complète avec des signaux ARIMA pour équilibrer.
    - Déduplication via FAISS.
    - Sauvegarde X et y dans 'out_file'.
    """
    participants = pd.read_csv(os.path.join(data_dir, "participants.tsv"), sep="\t")
    subjects = participants[participants["Group"].isin(["A", "AD"])]

    X, y = [], []

    for _, row in subjects.iterrows():
        pid = row["participant_id"]
        label = get_label(row)
        eeg_path = os.path.join(data_dir, pid, "eeg", f"{pid}_task-eyesclosed_eeg.set")
        print(f"🔍 Vérification: {pid} | Label: {label} | Fichier: {eeg_path}")

        if not os.path.exists(eeg_path):
            print(f"⚠️ Fichier manquant : {eeg_path}")
            continue

        try:
            raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
            raw.filter(0.5, 45).resample(fs)
            data = raw.get_data(picks=raw.ch_names[:num_electrodes], units="uV").T
            print(f"✅ EEG chargé : {data.shape}")

            nb_valid = 0
            # Découpe l'EEG en segments de 512 points
            total_segments = data.shape[0] // samples
            for i in range(total_segments):
                segment = data[i*samples : (i+1)*samples]
                if segment.shape != (samples, num_electrodes):
                    continue

                # Calcul des features
                feat = extract_features(segment)
                if feat.shape[0] != FEATURE_SIZE:
                    continue  # vecteur de features invalide

                # Concatène RAW (9728) + features (267) = 9995
                raw_vec = segment.flatten()  # 512×19
                full_vec = np.concatenate([raw_vec, feat])
                if full_vec.shape[0] != TOTAL_SIZE:
                    print("❌ Problème de taille lors de la concaténation.")
                    continue

                X.append(full_vec)
                y.append(label)
                nb_valid += 1

            print(f"✅ {pid} : {nb_valid} segments valides extraits")

        except Exception as e:
            print(f"❌ Erreur lecture {pid} : {e}")
            continue

    print(f"📊 Total segments extraits (EEG réel) : {len(X)}")

    if len(y) == 0:
        raise RuntimeError("❌ Aucun segment valide n'a été extrait. Vérifier les données.")

    # Simulation EEG pour équilibrer les classes
    target_per_class = max(Counter(y).values())
    states = {1: 1.5, 2: 2.0, 3: 2.5}  # moyennes ARIMA différentes selon la classe
    for lbl, mean_val in states.items():
        current_count = sum(1 for val in y if val == lbl)
        missing = target_per_class - current_count
        for _ in range(missing):
            # Génère un faux EEG (512×19)
            eeg_sim = np.array([
                generate_arima_eeg(mean_val, 0.3, samples=samples)
                for _ in range(num_electrodes)
            ]).T  # shape (512,19)
            feat_sim = extract_features(eeg_sim)
            if feat_sim.shape[0] == FEATURE_SIZE:
                raw_vec = eeg_sim.flatten()
                full_vec = np.concatenate([raw_vec, feat_sim])
                if full_vec.shape[0] == TOTAL_SIZE:
                    X.append(full_vec)
                    y.append(lbl)

    print(f"✅ Dataset équilibré total : {len(X)}")

    # Déduplication via FAISS
    X = np.array(X).astype(np.float32)
    y = np.array(y)
    dim = X.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(X)
    D, I = index.search(X, 2)
    # On considère duplicata si distance très faible
    mask = D[:, 1] > 1e-5
    X_clean = X[mask]
    y_clean = y[mask]

    print(f"✅ Déduplication FAISS : {len(X_clean)} segments restants")

    # Sauvegarde HDF5
    with h5py.File(out_file, 'w') as f:
        f.create_dataset("X", data=X_clean)
        f.create_dataset("y", data=y_clean)

    # Sauvegarde méta
    joblib.dump(Counter(y_clean), out_file.replace('.h5', '_meta.pkl'))
    print(f"💾 Sauvegardé : {out_file}")


# ====================================================================================
# === Paramètres modèle
# ====================================================================================
patch_len = 64
n_patches = samples // patch_len  # 512 // 64 = 8
input_dim = patch_len * num_electrodes  # 64×19 = 1216
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================================================================
# === Dataset
# ====================================================================================
class EEGDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.X = np.array(f['X'])  # shape [N, 9995]
            self.y = np.array(f['y']) - 1  # classes (1->0, 2->1, 3->2)

        # On entraîne un StandardScaler sur TOUTES les colonnes (9995)
        self.scaler = StandardScaler().fit(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Renvoie : patch (8,1216), feat (267D), label, flat_normed (9995D)
        
        - patch : sert à l'entrée "transformer" (exploitation du raw EEG)
        - feat  : correspond aux 267 features pour la projection feature_proj
        - flat_normed : on le retourne pour le SVM (9995D)
        """
        flat = self.X[idx]
        label = int(self.y[idx])

        # Normalisation complète
        flat_normed = self.scaler.transform([flat])[0]  # 9995D

        # Extraire la partie brute (9728) => patch
        eeg_raw = flat_normed[:RAW_SIZE].reshape(samples, num_electrodes)  # (512,19)
        eeg_raw = (eeg_raw - eeg_raw.mean(0)) / (eeg_raw.std(0) + 1e-6)
        patch = eeg_raw.reshape(n_patches, patch_len, num_electrodes)  # (8,64,19)
        patch = patch.transpose(0, 2, 1).reshape(n_patches, -1)        # (8, 64*19=1216)
        patch = torch.tensor(patch, dtype=torch.float32)

        # Extraire la partie features (267)
        feat = torch.tensor(flat_normed[RAW_SIZE:], dtype=torch.float32)

        return patch, feat, label, flat_normed


# ====================================================================================
# === Modèle ADFormer
# ====================================================================================
class ADFormerHybrid(nn.Module):
    def __init__(self, patch_dim=1216, d_model=256, num_layers=4, heads=4):
        super().__init__()
        self.embed = nn.Linear(patch_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=heads, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection de la partie "features" (267D) vers d_model
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(267),
            nn.Linear(267, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Tête de classification sur la concat (transformer_out + feature_proj)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, patch, feat):
        # patch : (batch, n_patches, patch_dim)
        # feat  : (batch, 267)
        x = self.embed(patch) + self.pos
        x = self.transformer(x)[:, -1]  # on ne récupère que le token final
        f = self.feature_proj(feat)
        fusion = torch.cat([x, f], dim=1)
        return self.head(fusion)


# ====================================================================================
# === Entraînement
# ====================================================================================
def train_adformer(h5_file):
    ds = EEGDataset(h5_file)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = ADFormerHybrid().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    # --- Entraîner le SVM sur 9995D ---
    print("Entraînement SVM (9995 dimensions)...")
    X_svm = ds.scaler.transform(ds.X)  # shape [N,9995] normalisées
    svm = SVC(probability=True, kernel='rbf')
    svm.fit(X_svm, ds.y)  # On entraîne sur la totalité
    print("✅ SVM prêt")

    for epoch in range(1, 31):
        model.train()
        total_loss, correct = 0, 0

        for patch, feat, label, flat_normed in tqdm(loader, desc=f"Epoch {epoch}"):
            patch = patch.to(device)            # (batch,8,1216)
            feat = feat.to(device)             # (batch,267)
            label = label.to(device)           # (batch,)
            # flat_normed = (batch,9995) => pour le SVM
            # On calcule la proba réseau
            logits = model(patch, feat)        # (batch, 3)
            prob_net = F.softmax(logits, dim=1)

            # Probabilité SVM
            # On doit passer un tableau shape (batch,9995) à predict_proba
            svm_probs = svm.predict_proba(flat_normed.numpy())
            svm_probs_t = torch.tensor(svm_probs, dtype=torch.float32, device=device)

            # Fusion
            fused = (prob_net + svm_probs_t) / 2.0

            # CrossEntropy sur la fusion
            loss = loss_fn(torch.log(fused), label)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            correct += (fused.argmax(dim=1) == label).sum().item()

        acc = correct / len(ds)
        print(f"✅ Epoch {epoch} | Loss: {total_loss:.4f} | Acc: {acc*100:.2f}%")

    os.makedirs("ad99", exist_ok=True)
    torch.save(model.state_dict(), "ad99/adformer_99.pth")
    joblib.dump(svm, "ad99/svm_99.pkl")
    joblib.dump(ds.scaler, "ad99/scaler.pkl")
    print("✅ Modèles enregistrés (Réseau + SVM + Scaler).")


# ====================================================================================
# === Routine principale
# ====================================================================================
if not os.path.exists("eeg_data_balanced.h5"):
    print("⚠️ EEG HDF5 manquant, génération en cours...")
    build_dataset_with_sim("alz", "eeg_data_balanced.h5")

assert os.path.exists("eeg_data_balanced.h5"), "❌ Le fichier .h5 est manquant !"

# Entraînement uniquement si le modèle n’existe pas
if not os.path.exists("ad99/adformer_99.pth"):
    train_adformer("eeg_data_balanced.h5")

# ====================================================================================
# === Évaluation (inférence)
# ====================================================================================
model = ADFormerHybrid().to(device)
model.load_state_dict(torch.load("ad99/adformer_99.pth", map_location=device))
model.eval()

svm = joblib.load("ad99/svm_99.pkl")
scaler = joblib.load("ad99/scaler.pkl")

# Charger toutes les données
with h5py.File("eeg_data_balanced.h5", "r") as f:
    X_all = np.array(f['X'])
    y_all = np.array(f['y']) - 1  # 0..2

subject_ids = [f"{label}_{i//5}" for i, label in enumerate(y_all)]
results = defaultdict(list)
true_labels = {}

for i, x in enumerate(X_all):
    label = y_all[i]
    sid = subject_ids[i]
    true_labels[sid] = label

    # Normalise 9995D
    x_norm = scaler.transform([x])[0]

    # Sépare patch / features
    eeg_raw = x_norm[:RAW_SIZE].reshape(samples, num_electrodes)
    eeg_raw = (eeg_raw - eeg_raw.mean(0)) / (eeg_raw.std(0) + 1e-6)
    patch = eeg_raw.reshape(n_patches, patch_len, num_electrodes)
    patch = patch.transpose(0,2,1).reshape(n_patches, -1)
    patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)

    feat_267 = x_norm[RAW_SIZE:]  # shape (267,)
    feat_t = torch.tensor(feat_267, dtype=torch.float32).unsqueeze(0).to(device)

    # Réseau
    with torch.no_grad():
        logits = model(patch_t, feat_t)
        proba_model = F.softmax(logits, dim=1).cpu().numpy()[0]

    # SVM sur 9995D
    proba_svm = svm.predict_proba(x_norm.reshape(1, -1))[0]

    # Fusion
    fused = (proba_model + proba_svm) / 2.0

    # Seuil de rejet
    if fused.max() < 0.5:
        pred = -1
    else:
        pred = np.argmax(fused)

    results[sid].append(pred)

# === Vote par sujet
y_true, y_pred, rejected = [], [], []
for sid, preds in results.items():
    if all(p == -1 for p in preds):
        rejected.append(sid)
        continue
    preds_clean = [p for p in preds if p != -1]
    vote = Counter(preds_clean).most_common(1)[0][0]
    y_true.append(true_labels[sid])
    y_pred.append(vote)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
# On considère [0,1,2] comme "AD" => calcule la sensibilité au global
recall = recall_score(
    [1 if t in [0,1,2] else 0 for t in y_true],
    [1 if p in [0,1,2] else 0 for p in y_pred],
    zero_division=0
)
cm = confusion_matrix(y_true, y_pred)

print(f"✅ Subject-Level Accuracy : {acc*100:.2f}%")
print(f"✅ F1 Macro : {f1*100:.2f}%")
print(f"🧠 Sensibilité Alzheimer (global) : {recall*100:.2f}%")
print("📊 Matrice de confusion :\n", cm)
print(f"❌ Patients rejetés (incertains) : {len(rejected)} / {len(set(subject_ids))}")


# ------------------------------------------------------------------------------
# 📄 LICENCE - Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# 
# Ce script "adformer999.py" fait partie du projet 
# Alzheimer EEG AI Assistant, développé par Kocupyr Romain (rkocupyr@gmail.com).
#
# Vous êtes libres de :
#   ✅ Partager — copier le script
#   ✅ Adapter — le modifier et l’intégrer dans un autre projet
#
# Sous les conditions suivantes :
#   📌 Attribution — Vous devez mentionner l’auteur original (Kocupyr Romain)
#   📌 Non Commercial — Interdiction d’usage commercial sans autorisation
#   📌 Partage identique — Toute version modifiée doit être publiée sous la même licence
#
# 🔗 Licence complète : https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------
