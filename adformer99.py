"""
adformer99.py
Licence : Creative Commons BY-NC-SA 4.0
Auteurs : 
    - Kocupyr Romain (créateur et chef de projet) : rkocupyr@gmail.com
    - dev = GPT multi_gpt_api (OpenAI)
    - Grok3
Dataset = https://www.kaggle.com/datasets/yosftag/open-nuro-dataset
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, 
    recall_score
)
from sklearn.svm import SVC

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

# Le modèle principal exploitera en interne un "feature_dim" = 267 pour la partie features
MODEL_FEATURE_DIM = FEATURE_SIZE

# Paires d’asymétrie utilisées
asym_pairs = [(3, 5), (13, 15), (0, 1)]

# Modèle de Kalman pour le lissage
kf_model = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

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

        # === PSD (Welch)
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


def get_labels(row):
    """
    Renvoie un tuple (y_main, y_stage).
    - y_main : classe binaire
        0 => non-AD (Group = 'A')
        1 => AD (Group = 'AD')
        -1 => si hors scope
    - y_stage : si AD, renvoie 1/2/3 selon la sévérité ; sinon -1
    """
    if row["Group"] not in ["A", "AD"]:
        return -1, -1  # Hors scope

    # Binaire
    y_main = 0 if row["Group"] == "A" else 1

    # Stade basé sur le MMSE (uniquement si AD)
    if y_main == 0:
        return y_main, -1

    mmse = row["MMSE"]
    if pd.isna(mmse):
        return y_main, 1
    elif mmse >= 19:
        return y_main, 1
    elif mmse >= 10:
        return y_main, 2
    else:
        return y_main, 3


def generate_arima_eeg(mean, std, samples=512):
    """
    Génère un signal EEG fictif par un processus ARIMA
    (ex: ar=1, ma=0.5) autour d'une moyenne et d'un écart-type donnés.
    """
    ar_params = np.array([1, -0.5])
    ma_params = np.array([1])
    return mean + std * ArmaProcess(ar_params, ma_params).generate_sample(nsample=samples)


def build_dataset_with_sim(data_dir, out_file):
    """
    Construit et sauvegarde en HDF5 un dataset équilibré (classe binaire AD vs non-AD).
    - Parcourt les EEG existants, extrait tous les segments valides (512×19).
    - Calcule 267 features.
    - Concatène (raw + features) -> vecteur de taille 9995.
    - Stocke y_main et y_stage.
    - Complète avec des signaux ARIMA pour équilibrer AD vs non-AD.
    - Déduplication via FAISS.
    - Sauvegarde X, y_main et y_stage dans 'out_file'.
    """
    participants = pd.read_csv(os.path.join(data_dir, "participants.tsv"), sep="\t")
    subjects = participants[participants["Group"].isin(["A", "AD"])]

    X, y_main_list, y_stage_list = [], [], []

    for _, row in subjects.iterrows():
        pid = row["participant_id"]
        y_main, y_stage = get_labels(row)
        if y_main == -1:
            continue  # hors scope

        eeg_path = os.path.join(data_dir, pid, "eeg", f"{pid}_task-eyesclosed_eeg.set")
        print(f"🔍 Vérification: {pid} | y_main={y_main} | y_stage={y_stage} | Fichier: {eeg_path}")

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
                    # vecteur de features invalide
                    continue

                # Concatène RAW (9728) + features (267) = 9995
                raw_vec = segment.flatten()  # 512×19
                full_vec = np.concatenate([raw_vec, feat])
                if full_vec.shape[0] != TOTAL_SIZE:
                    print("❌ Problème de taille lors de la concaténation.")
                    continue

                X.append(full_vec)
                y_main_list.append(y_main)
                y_stage_list.append(y_stage)
                nb_valid += 1

            print(f"✅ {pid} : {nb_valid} segments valides extraits")

        except Exception as e:
            print(f"❌ Erreur lecture {pid} : {e}")
            continue

    print(f"📊 Total segments extraits (EEG réel) : {len(X)}")

    if len(X) == 0:
        raise RuntimeError("❌ Aucun segment valide n'a été extrait. Vérifier les données.")

    # Équilibrage binaire via simulation EEG (ARIMA)
    counter_main = Counter(y_main_list)
    # On ne traite que y_main=0 ou 1, donc on récupère la classe majoritaire
    target_per_class = max(counter_main.values())
    print("Équilibrage binaire AD vs non-AD...")
    for label_bin in [0, 1]:
        current_count = counter_main.get(label_bin, 0)
        missing = target_per_class - current_count
        if missing > 0:
            # On génère des segments artificiels
            # Choix d'une moyenne distincte pour chaque classe
            mean_val = 0.5 if label_bin == 0 else 2.0
            for _ in range(missing):
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
                        y_main_list.append(label_bin)
                        # Pour la simulation, on ne peut pas vraiment estimer le vrai stade
                        # => on met y_stage=-1 si label_bin=0 (non-AD)
                        #    ou un stade "fictif" (ex: 2) si label_bin=1
                        if label_bin == 1:
                            y_stage_list.append(2)  # stade moyen fictif
                        else:
                            y_stage_list.append(-1)

    print(f"✅ Dataset équilibré total (binaire) : {len(X)}")

    # Déduplication via FAISS
    X = np.array(X).astype(np.float32)
    y_main_array = np.array(y_main_list)
    y_stage_array = np.array(y_stage_list)

    dim = X.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(X)
    D, I = index.search(X, 2)
    # On considère duplicata si distance très faible
    mask = D[:, 1] > 1e-5
    X_clean = X[mask]
    y_main_clean = y_main_array[mask]
    y_stage_clean = y_stage_array[mask]

    print(f"✅ Déduplication FAISS : {len(X_clean)} segments restants")

    # Sauvegarde HDF5
    with h5py.File(out_file, 'w') as f:
        f.create_dataset("X", data=X_clean)
        f.create_dataset("y_main", data=y_main_clean)
        f.create_dataset("y_stage", data=y_stage_clean)

    # Sauvegarde méta
    meta_info = {
        'counter_main': dict(Counter(y_main_clean)),
        'counter_stage': dict(Counter(y_stage_clean))
    }
    joblib.dump(meta_info, out_file.replace('.h5', '_meta.pkl'))
    print(f"💾 Sauvegardé : {out_file}")


# === Hyperparamètres du pipeline binaire
patch_len = 64
n_patches = samples // patch_len  # 512 // 64 = 8
input_dim = patch_len * num_electrodes  # 64×19 = 1216
feature_dim = FEATURE_SIZE        # = 267
num_classes_binary = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Dataset pour la classification binaire
class EEGDatasetBinary(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.X = np.array(f['X'])           # shape [N, 9995]
            self.y_main = np.array(f['y_main']) # 0 ou 1
            self.y_stage = np.array(f['y_stage'])
        # StandardScaler sur TOUTES les colonnes
        self.scaler = StandardScaler().fit(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Renvoie (patches, features, label_binaire).
        - patches : (n_patches, input_dim) -> (8, 1216)
        - features : dimension 267
        - label : 0 ou 1
        """
        flat = self.X[idx]
        label = int(self.y_main[idx])

        # Normalisation sur l'ensemble du vecteur avant split
        flat_normed = self.scaler.transform([flat])[0]

        # Récupération EEG brut (512×19 = 9728)
        eeg_raw = flat_normed[:RAW_SIZE].reshape(samples, num_electrodes)
        eeg_raw = (eeg_raw - eeg_raw.mean(0)) / (eeg_raw.std(0) + 1e-6)
        patch = eeg_raw.reshape(n_patches, patch_len, num_electrodes)
        patch = patch.transpose(0, 2, 1).reshape(n_patches, -1)
        patch_t = torch.tensor(patch, dtype=torch.float32)

        # Récupération des 267 features
        feat_t = torch.tensor(flat_normed[RAW_SIZE:], dtype=torch.float32)

        return patch_t, feat_t, torch.tensor(label, dtype=torch.long)


# === Modèle ADFormer pour la classification binaire AD vs non-AD
class ADFormerBinary(nn.Module):
    def __init__(self, patch_dim=input_dim, d_model=256, num_layers=4, heads=4):
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
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Tête de classification (2 classes)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes_binary)
        )

    def forward(self, patch, feat):
        # patch : (batch, n_patches, patch_dim)
        # feat : (batch, 267)
        x = self.embed(patch) + self.pos
        x = self.transformer(x)[:, -1]  # on ne récupère que le token final
        f = self.feature_proj(feat)
        fusion = torch.cat([x, f], dim=1)
        return self.head(fusion)


def train_adformer_binary(h5_file):
    """
    Entraîne le modèle ADFormerBinary + SVM en fusion de prédiction pour la classification binaire.
    """
    ds = EEGDatasetBinary(h5_file)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = ADFormerBinary().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Entraînement du SVM UNIQUEMENT sur les 267 features
    X_svm = ds.X[:, RAW_SIZE:]  # shape [N, 267]
    y_svm = ds.y_main
    svm = SVC(probability=True, kernel='rbf')
    print("Entraînement SVM (267 dims) [Binaire]...")
    svm.fit(X_svm, y_svm)
    print("✅ SVM binaire prêt")

    for epoch in range(1, 6):
        model.train()
        total_loss, correct = 0, 0
        for patch, feat, y in tqdm(loader, desc=f"[Binaire] Epoch {epoch}"):
            patch, feat, y = patch.to(device), feat.to(device), y.to(device)
            logits = model(patch, feat)

            # Fusion des prédictions SVM et du réseau
            prob_net = F.softmax(logits, dim=1)
            svm_probs = svm.predict_proba(feat.cpu().numpy())
            svm_probs_t = torch.tensor(svm_probs, device=device, dtype=torch.float32)
            # Weighted fusion simple
            fused = (prob_net + svm_probs_t) / 2

            loss = loss_fn(torch.log(fused), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            correct += (fused.argmax(dim=1) == y).sum().item()

        acc = correct / len(ds)
        print(f"✅ Epoch {epoch} | Loss: {total_loss:.4f} | Acc: {acc*100:.2f}%")

    os.makedirs("ad_models", exist_ok=True)
    torch.save(model.state_dict(), "ad_models/adformer_binary.pth")
    joblib.dump(svm, "ad_models/svm_binary.pkl")
    joblib.dump(ds.scaler, "ad_models/scaler_binary.pkl")
    print("✅ Modèles enregistrés (Réseau binaire + SVM + Scaler).")


# === Pipeline secondaire (Staging) =============================================

class EEGDatasetStaging(Dataset):
    """
    Dataset pour la classification du stade (1/2/3) UNIQUEMENT pour les segments AD.
    Utilise seulement les 267 features pour un réseau plus simple.
    """
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            X_full = np.array(f['X'])
            y_main = np.array(f['y_main'])
            y_stage = np.array(f['y_stage'])

        # On filtre : ne garder que ceux dont y_main=1 (AD) et y_stage in [1,2,3]
        mask_ad = (y_main == 1) & (y_stage > 0)
        self.X_feat = X_full[mask_ad, RAW_SIZE:]  # ne garde que la partie features
        self.y_stage = y_stage[mask_ad] - 1       # 0,1,2 pour [1,2,3]

        self.scaler = StandardScaler().fit(self.X_feat)

    def __len__(self):
        return len(self.X_feat)

    def __getitem__(self, idx):
        xfeat = self.X_feat[idx]
        y = self.y_stage[idx]
        x_norm = self.scaler.transform([xfeat])[0]
        return torch.tensor(x_norm, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class StageNet(nn.Module):
    """
    Réseau MLP simple pour classer 3 stades (1/2/3 => 0/1/2).
    Entrée : 267 features
    Sortie : 3 classes
    """
    def __init__(self, input_dim=267, hidden=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)


def train_stage_network(h5_file):
    """
    Entraîne le MLP (StageNet) pour la classification de stade 1/2/3 chez les AD.
    """
    ds_stage = EEGDatasetStaging(h5_file)
    loader_stage = DataLoader(ds_stage, batch_size=64, shuffle=True)

    model_s = StageNet().to(device)
    opt = torch.optim.Adam(model_s.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 6):
        model_s.train()
        total_loss, correct = 0, 0
        for xfeat, y in tqdm(loader_stage, desc=f"[Staging] Epoch {epoch}"):
            xfeat, y = xfeat.to(device), y.to(device)
            logits = model_s(xfeat)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()

        acc = correct / len(ds_stage)
        print(f"✅ [Staging] Epoch {epoch} | Loss: {total_loss:.4f} | Acc: {acc*100:.2f}%")

    torch.save(model_s.state_dict(), "ad_models/stagenet.pth")
    joblib.dump(ds_stage.scaler, "ad_models/scaler_stage.pkl")
    print("✅ Modèle de staging enregistré (StageNet + scaler).")


# ------------------------------------------------------------------------------
# Exécution principale
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Construction du dataset HDF5 s'il n'existe pas
    if not os.path.exists("eeg_data_balanced.h5"):
        print("⚠️ EEG HDF5 manquant, génération en cours...")
        build_dataset_with_sim("alz", "eeg_data_balanced.h5")
    assert os.path.exists("eeg_data_balanced.h5"), "❌ Le fichier .h5 est manquant !"

    # 2) Entraînement binaire si le modèle n’existe pas
    if not os.path.exists("ad_models/adformer_binary.pth"):
        train_adformer_binary("eeg_data_balanced.h5")

    # 3) Entraînement staging si pas de modèle existant
    if not os.path.exists("ad_models/stagenet.pth"):
        train_stage_network("eeg_data_balanced.h5")

    # =========================
    # Exemple d'inférence
    # =========================
    print("\n=== Démonstration d'inférence sur l'ensemble du dataset ===")

    # On recharge le modèle binaire
    model_bin = ADFormerBinary().to(device)
    model_bin.load_state_dict(torch.load("ad_models/adformer_binary.pth", map_location=device))
    model_bin.eval()

    svm_bin = joblib.load("ad_models/svm_binary.pkl")
    scaler_bin = joblib.load("ad_models/scaler_binary.pkl")

    # On recharge le modèle de staging
    model_stage = StageNet().to(device)
    model_stage.load_state_dict(torch.load("ad_models/stagenet.pth", map_location=device))
    model_stage.eval()

    scaler_stage = joblib.load("ad_models/scaler_stage.pkl")

    # Chargement de toutes les données pour test
    with h5py.File("eeg_data_balanced.h5", "r") as f:
        X_all = np.array(f['X'])
        y_main_all = np.array(f['y_main'])
        y_stage_all = np.array(f['y_stage'])

    # On simule un "vote" au niveau sujet. On crée des IDs factices.
    subject_ids = [f"{y_main}_{i//5}" for i, y_main in enumerate(y_main_all)]
    results_main = defaultdict(list)
    results_stage = defaultdict(list)

    for i, x in enumerate(X_all):
        sid = subject_ids[i]
        true_main = y_main_all[i]  # 0 ou 1
        true_stage = y_stage_all[i]  # 1,2,3 ou -1

        # =====================
        # 1) Classification binaire
        # =====================
        # Normalisation (binaire)
        x_norm = scaler_bin.transform([x])[0]

        # Extraction patch + feat
        eeg_raw = x_norm[:RAW_SIZE].reshape(samples, num_electrodes)
        eeg_raw = (eeg_raw - eeg_raw.mean(0)) / (eeg_raw.std(0) + 1e-6)
        patch = eeg_raw.reshape(n_patches, patch_len, num_electrodes)
        patch = patch.transpose(0,2,1).reshape(n_patches, -1)
        patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)

        feat_bin = x_norm[RAW_SIZE:]
        feat_bin_t = torch.tensor(feat_bin, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_bin = model_bin(patch_t, feat_bin_t)
            prob_net_bin = F.softmax(logits_bin, dim=1).cpu().numpy()[0]
            prob_svm_bin = svm_bin.predict_proba(feat_bin.reshape(1, -1))[0]
            fused_bin = (prob_net_bin + prob_svm_bin) / 2
        pred_bin = np.argmax(fused_bin)

        results_main[sid].append(pred_bin)

        # =====================
        # 2) Staging (si AD)
        # =====================
        if pred_bin == 1:
            # Normalisation staging
            x_feat_stage = x[RAW_SIZE:]  # on prend directement les features bruts
            x_feat_stage_norm = scaler_stage.transform([x_feat_stage])[0]
            x_feat_stage_t = torch.tensor(x_feat_stage_norm, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits_stage = model_stage(x_feat_stage_t)
                pred_stage = logits_stage.argmax(dim=1).item()  # 0/1/2 => stade 1/2/3
            results_stage[sid].append(pred_stage)
        else:
            results_stage[sid].append(-1)  # pas AD => pas de stade


    # Vote par sujet
    final_main_true, final_main_pred = [], []
    final_stage_true, final_stage_pred = [], []
    rejected = []

    for sid, preds_main in results_main.items():
        # Binaire
        vote_main = Counter(preds_main).most_common(1)[0][0]
        # Vrai label
        index_ex = int(sid.split("_")[-1]) * 5  # petit bidouillage, non critique
        true_m = y_main_all[index_ex]
        final_main_true.append(true_m)
        final_main_pred.append(vote_main)

        # Staging
        if vote_main == 1:
            stage_preds = results_stage[sid]
            # si tout est -1 => rejet
            if all(s == -1 for s in stage_preds):
                rejected.append(sid)
                st_vote = -1
            else:
                # On retire les -1 éventuels
                st_clean = [s for s in stage_preds if s != -1]
                st_vote = Counter(st_clean).most_common(1)[0][0]
            true_st = y_stage_all[index_ex]
            final_stage_true.append(true_st)
            final_stage_pred.append(st_vote)
        else:
            # Non-AD => on ne calcule pas de stade
            final_stage_true.append(-1)
            final_stage_pred.append(-1)

    # Évaluation Binaire
    acc_bin = accuracy_score(final_main_true, final_main_pred)
    f1_bin = f1_score(final_main_true, final_main_pred, average="macro")
    cm_bin = confusion_matrix(final_main_true, final_main_pred)
    print("\n=== Résultats Classification Binaire (Sujet) ===")
    print(f"Accuracy binaire : {acc_bin*100:.2f}%")
    print(f"F1 Macro binaire : {f1_bin*100:.2f}%")
    print("Matrice de confusion binaire :\n", cm_bin)

    # Évaluation Staging (sur les sujets AD)
    mask_ad_eval = [t == 1 for t in final_main_true]  # y_main=1
    true_stages = np.array(final_stage_true)[mask_ad_eval]
    pred_stages = np.array(final_stage_pred)[mask_ad_eval]

    # On enlève les -1 (rejet) si besoin
    valid_idx = (pred_stages != -1)
    if valid_idx.sum() > 0:
        st_acc = accuracy_score(true_stages[valid_idx], pred_stages[valid_idx])
        st_f1 = f1_score(true_stages[valid_idx], pred_stages[valid_idx], average="macro")
        cm_st = confusion_matrix(true_stages[valid_idx], pred_stages[valid_idx])
        print("\n=== Résultats Staging (Sujet AD) ===")
        print(f"Accuracy staging : {st_acc*100:.2f}%")
        print(f"F1 Macro staging : {st_f1*100:.2f}%")
        print("Matrice de confusion staging :\n", cm_st)
    else:
        print("Aucun sujet AD n'a été classé pour le staging ou tous rejetés.")

    print(f"❌ Nombre de sujets rejetés (staging) : {len(rejected)}")

# ------------------------------------------------------------------------------
# 📄 LICENCE - Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# 
# Ce script "adformer99.py" fait partie du projet 
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
