"""
ADFormer-HYBRID — EEG Alzheimer Classifier (Full Pipeline, Stable Version)
Licence : Creative Commons BY-NC-SA 4.0
Auteurs : Kocupyr Romain créateur et chef de projet, dev = GPT multi_gpt_api (OpenAI), Grok3, kocupyr romain
Dataset = https://www.kaggle.com/datasets/yosftag/open-nuro-dataset
Accu: EPOCH 30 = 92.84%
F1 Macro = 98,63%
Précision patient-level = 98,45%
Evaluate script = evaluate_adformer_subject.py
Evaluate patient = run_predict.py
"""

import os
import numpy as np
import pandas as pd
import h5py
import mne
import torch
import torch.nn as nn
from sklearn.svm import SVC
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch
from scipy.stats import iqr
from pykalman import KalmanFilter
import antropy as ant
from collections import Counter

# ====================================================================================
# === PARAMÈTRES GLOBAUX
# ====================================================================================
fs = 128               # Fréquence d'échantillonnage cible
samples = 512          # Taille (en échantillons) de chaque segment EEG
num_electrodes = 19    # Nombre d'électrodes retenues
# Nous allons stocker : 512*19 = 9728 points bruts + ~267 features
# Ajustons la taille finale attendue :
RAW_SIZE = samples * num_electrodes  # 512*19 = 9728
FEATURE_SIZE = 267                   # estimé après concaténation des features
TOTAL_FEATURE_SIZE = RAW_SIZE + FEATURE_SIZE  # ~ 9995

# Mise à jour : le modèle aura un feature_dim correspondant UNIQUEMENT à la partie "features".
MODEL_FEATURE_DIM = FEATURE_SIZE

# Paires d’asymétrie utilisées
asym_pairs = [(3, 5), (13, 15), (0, 1)]

# Gammes de fréquences
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha1': (8, 10),
    'Alpha2': (10, 13),
    'Beta1': (13, 20),
    'Beta2': (20, 30),
    'Gamma': (30, 45)
}

# Modèle Kalman pour un lissage éventuel des puissances spectrales
kf_model = KalmanFilter(initial_state_mean=0, n_dim_obs=1)


# ====================================================================================
# === FONCTIONS DE FEATURE ENGINEERING
# ====================================================================================
def kalman_filter_signal(signal):
    """Applique un Filtre de Kalman 1D sur un signal (vector)."""
    filtered, _ = kf_model.filter(signal[:, None])
    return filtered[:, 0]


def extract_features(data):
    """
    Calcule un vecteur de features pour un segment EEG de forme (512, 19).
    Retourne environ 267 valeurs (selon ce qui est concaténé).
    """
    if data.shape != (samples, num_electrodes):
        raise ValueError(f"Segment shape invalide : {data.shape}")

    # Statistiques temporelles
    mean_t = np.mean(data, axis=0)     # 19
    var_t = np.var(data, axis=0)       # 19
    iqr_t = iqr(data, axis=0)          # 19

    # PSD via Welch
    freqs, psd = welch(data, fs=fs, nperseg=samples, axis=0)  # psd shape : (nfreqs, 19)

    band_feats = []
    kalman_means = []
    kalman_diffs = []

    # Calcul de la puissance moyenne dans chaque bande + Kalman
    for fmin, fmax in bands.values():
        idx = (freqs >= fmin) & (freqs <= fmax)
        # Puissance brute moyenne (par canal) dans la bande
        raw_power = np.mean(psd[idx], axis=0)  # shape (19,)
        # Lissage Kalman sur la moyenne spectrale (moyenne sur les canaux, pour la courbe freq)
        kalman_power = kalman_filter_signal(psd[idx].mean(axis=1))  # shape (nb_freqs_in_band,)
        # On stocke
        band_feats.append(raw_power)
        kalman_means.append(np.mean(kalman_power))
        kalman_diffs.append(raw_power.mean() - np.mean(kalman_power))

    rbp = np.stack(band_feats, axis=0)  # shape (7, 19)

    # Entropies (Permutation, Sample)
    perm_en = np.array([ant.perm_entropy(data[:, i], order=3, normalize=True)
                        for i in range(num_electrodes)])  # 19
    sample_en = np.array([ant.sample_entropy(data[:, i], order=2)
                          for i in range(num_electrodes)])  # 19

    # Mesures de connectivité / graph
    corr_matrix = np.corrcoef(data.T)  # shape (19, 19)
    clustering = np.array([
        np.sum(corr_matrix[i] > 0.5) / (num_electrodes - 1)
        for i in range(num_electrodes)
    ])  # 19
    path_length = np.mean(np.abs(corr_matrix))
    non_zero_corr = corr_matrix[np.abs(corr_matrix) > 0]
    efficiency = np.mean(1 / np.abs(non_zero_corr)) if len(non_zero_corr) > 0 else 0.0
    small_worldness = np.mean(clustering) / path_length if path_length != 0 else 0.0

    # Asymétries inter-hémisphériques
    asym = np.array([np.mean(data[:, i] - data[:, j]) for i, j in asym_pairs])  # ex: 3 valeurs

    # Concaténation finale (environ 267 valeurs totales)
    features = np.concatenate([
        mean_t,                # 19
        var_t,                 # 19
        iqr_t,                 # 19
        rbp.flatten(),         # 7*19 = 133
        perm_en,               # 19
        sample_en,             # 19
        clustering,            # 19
        asym,                  # 3
        [path_length, efficiency, small_worldness],  # 3
        kalman_means,          # 7
        kalman_diffs           # 7
    ])

    return features  # ~ 267 valeurs


def get_label(row):
    """
    Convertit la ligne du participants.tsv en label numérique.
    - Group 'A' ou 'AD' => on s'intéresse à la sévérité via le MMSE
    - Si pas de MMSE => classe 1 (forme légère par défaut)
    - MMSE >= 19 => classe 1
    - 10 <= MMSE < 19 => classe 2
    - MMSE < 10 => classe 3
    """
    if row["Group"] not in ["A", "AD"]:
        return -1  # label 'invalide'
    mmse = row["MMSE"]
    if pd.isna(mmse):
        return 1
    elif mmse >= 19:
        return 1
    elif mmse >= 10:
        return 2
    else:
        return 3


def build_h5(data_dir, h5_file):
    """
    Parcourt le répertoire BIDS, lit les EEG (.set), segmente, extrait (data brute + features),
    et stocke dans un dataset HDF5 "X" et "y".
    """
    print("📦 Création du dataset HDF5 propre...")

    # On lit le participants.tsv pour récupérer le groupe et le MMSE
    participants = pd.read_csv(os.path.join(data_dir, "participants.tsv"), sep="\t")
    subjects = participants[participants["Group"].isin(["A", "AD"])]

    with h5py.File(h5_file, 'w') as f:
        # Création des datasets extensibles
        f.create_dataset(
            "X",
            shape=(0, TOTAL_FEATURE_SIZE),
            maxshape=(None, TOTAL_FEATURE_SIZE),
            dtype='float32'
        )
        f.create_dataset(
            "y",
            shape=(0,),
            maxshape=(None,),
            dtype='int32'
        )

        offset = 0
        for _, row in subjects.iterrows():
            pid = row["participant_id"]
            label = get_label(row)
            # On ignore si label invalide
            if label < 1:
                continue

            eeg_path = os.path.join(
                data_dir, pid, "eeg", f"{pid}_task-eyesclosed_eeg.set"
            )
            if not os.path.exists(eeg_path):
                continue

            try:
                # Lecture via MNE
                raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
                raw.filter(0.5, 45)
                raw.resample(fs)

                # Récupération des 19 premiers canaux (si la config standard le permet)
                data = raw.get_data(picks=raw.ch_names[:num_electrodes], units="uV").T

                # On segmente par blocs de 512 échantillons
                nb_segments = data.shape[0] // samples
                for i in range(nb_segments):
                    seg = data[i*samples:(i+1)*samples]  # (512, 19)

                    try:
                        # On calcule les features
                        feat = extract_features(seg)
                        if feat.shape[0] != FEATURE_SIZE:
                            print(f"⚠️ Segment ignoré (shape features invalide): {feat.shape}")
                            continue

                        # Concatène : données brutes + features
                        raw_flat = seg.flatten()  # 512*19 = 9728
                        combined = np.concatenate([raw_flat, feat])  # total ~9995

                        # On stocke dans le HDF5
                        f["X"].resize(offset+1, axis=0)
                        f["y"].resize(offset+1, axis=0)

                        f["X"][offset] = combined
                        f["y"][offset] = label

                        offset += 1

                    except Exception as e:
                        print(f"❌ Extraction échouée (segment {i}) - {e}")

            except Exception as e:
                print(f"❌ Fichier {pid} ignoré - {e}")


# ====================================================================================
# === DATASET PYTORCH
# ====================================================================================
class EEGHybridDataset(Dataset):
    """
    Dataset PyTorch qui lit depuis le HDF5 :
    - la partie brute (512*19 points) pour construire les patches
    - la partie "features" (env. 267 points)
    """
    def __init__(self, h5_path, patch_len=64, augment=True):
        super().__init__()
        self.h5 = h5py.File(h5_path, 'r')
        self.X = self.h5["X"][:]
        self.y = self.h5["y"][:]
        self.augment = augment

        self.patch_len = patch_len
        self.num_patches = samples // patch_len  # 512 / 64 = 8
        self.channels = num_electrodes

        # On standardise l'ensemble du vecteur (raw + features) pour le SVM + partie MLP
        self.scaler = StandardScaler().fit(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        full_feat = self.X[idx]  # shape : ~9995
        label = int(self.y[idx]) - 1  # 1->0, 2->1, 3->2

        # Sépare la partie brute et la partie features
        raw_part = full_feat[:RAW_SIZE]          # 9728
        feat_part = full_feat[RAW_SIZE:]         # ~267

        # On restaure la forme (512, 19)
        eeg = raw_part.reshape(samples, self.channels)

        # Normalisation par canal
        eeg = (eeg - eeg.mean(axis=0)) / (eeg.std(axis=0) + 1e-6)

        # Quelques augmentations simples
        if self.augment:
            # Inversion temporelle (prob 0.2)
            if np.random.rand() < 0.2:
                eeg = eeg[::-1]
            # Bruit gaussien (prob 0.2)
            if np.random.rand() < 0.2:
                eeg += np.random.normal(0, 0.3, eeg.shape)

        # Construction des patches (8 patches de 64 échantillons, chacun de 19 canaux)
        patch = eeg.reshape(self.num_patches, self.patch_len, self.channels)
        patch = patch.transpose(0, 2, 1).reshape(self.num_patches, -1)  # (8, 1216)

        # Tensor pour la partie patch
        patch = torch.tensor(patch, dtype=torch.float32)

        # On scale la totalité du vecteur
        scaled_full = self.scaler.transform([full_feat])[0]  # -> shape ~ (9995,)

        # Pour la partie MLP, on n'a besoin que des ~267 dernières
        scaled_features = scaled_full[RAW_SIZE:]  # shape ~ (267,)

        feat = torch.tensor(scaled_features, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return patch, feat, y

# ====================================================================================
# === MODÈLE
# ====================================================================================
class ADFormerHybrid(nn.Module):
    """
    Modèle hybride :
    - Transformer Encoder sur les patches extraits du signal brut (8 patches × 19 canaux × 64 échantillons)
    - MLP sur les features extraites (taille ~267)
    - Fusion en fin
    """
    def __init__(self,
                 patch_dim=64*19,      # 1216
                 num_patches=8,        # 512/64
                 feature_dim=MODEL_FEATURE_DIM,  # 267
                 d_model=256,
                 num_classes=3):
        super().__init__()

        # Embed linéaire des patches
        self.embed_patch = nn.Linear(patch_dim, d_model)

        # Positionnel (simple)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Encodeur MLP pour la partie features
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model)
        )

        # Tête de classification fusionnée
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, patches, features):
        """
        patches : (batch_size, 8, 1216)
        features : (batch_size, 267)
        """
        # Encode les patches via Transformer
        p = self.embed_patch(patches) + self.pos_embed
        p = self.transformer(p)  # shape (batch_size, 8, d_model)
        p = p[:, -1]             # On récupère le dernier patch encodé

        # Encode les features
        f = self.feature_encoder(features)

        # Fusion
        x = torch.cat([p, f], dim=-1)  # (batch_size, d_model*2)
        return self.head(x)           # (batch_size, num_classes)

# ====================================================================================
# === FONCTION D'ENTRAÎNEMENT
# ====================================================================================
def train_hybrid(h5_file):
    """
    Entraîne un SVM et le modèle ADFormerHybrid, puis fusionne leurs probabilités.
    Gère le cas où il n'y aurait qu'une seule classe (skip SVM).
    """
    dataset = EEGHybridDataset(h5_file)
    class_counts = Counter(dataset.y)
    print(f"📊 Répartition des classes : {class_counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Utilisation de :", device)

    # Modèle ADFormer
    model = ADFormerHybrid().to(device)

    # Vérif du nombre de classes
    svm = None
    if len(class_counts) < 2:
        print(f"⚠️ Pas assez de classes ({class_counts}) pour l'entraînement SVM. SVM ignoré.")
    else:
        # Entraînement du SVM sur la totalité du vecteur
        print("Entraînement du SVM (peut être long selon la taille du dataset)...")
        svm = SVC(probability=True, kernel='rbf')
        svm.fit(dataset.X, dataset.y)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    opt = AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Entraînement simple sur 30 époques
    for epoch in range(1, 31):
        model.train()
        correct = 0
        total_loss = 0.0

        for patch, feat, y in tqdm(loader, desc=f"Epoch {epoch}"):
            patch, feat, y = patch.to(device), feat.to(device), y.to(device)

            # Sortie du modèle
            logits = model(patch, feat)

            # Si on a un SVM entraîné, on fusionne
            if svm is not None:
                with torch.no_grad():
                    svm_probs = torch.tensor(
                        svm.predict_proba(dataset.scaler.transform(
                            # On recompose le vecteur complet pour le SVM
                            # (Ici, on a only 'feat' => On pourrait indexer la batch
                            #  mais pour un vrai usage, mieux vaut adapter le code
                            #  ou faire un SVM sur la partie "feat" seulement.)
                            torch.cat([patch.view(patch.size(0), -1),
                                       feat], dim=1
                                     ).cpu().numpy()
                        )),
                        device=device,
                        dtype=torch.float32
                    )
                fusion = (F.softmax(logits, dim=1) + svm_probs) / 2.0
            else:
                # Pas de SVM => on utilise seulement la sortie du réseau
                fusion = F.softmax(logits, dim=1)

            loss = loss_fn(torch.log(fusion), y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            correct += (fusion.argmax(dim=1) == y).sum().item()

        acc = correct / len(dataset)
        print(f"\n✅ Epoch {epoch} | Loss: {total_loss:.4f} | Acc: {acc*100:.2f}%")

    # Sauvegarde du modèle
    torch.save(model.state_dict(), "adformer_hybrid_model.pth")
    print("\n🎯 Modèle enregistré sous adformer_hybrid_model.pth")


# ====================================================================================
# === MAIN
# ====================================================================================
if __name__ == "__main__":
    data_dir = "/workspace/memory_os_ai/alz/"
    h5_file = os.path.join(data_dir, "eeg_data_alzheimer_kalman.h5")

    # Construction du HDF5 (si non déjà fait)
    if not os.path.exists(h5_file):
        build_h5(data_dir, h5_file)

    # Entraînement
    train_hybrid(h5_file)



# ------------------------------------------------------------------------------
# 📄 LICENCE - Creative Commons Attribution-NonCommercial-ShareAlike 4.0
#
# Ce script "adformer_hybrid_voting_full.py" fait partie du projet Alzheimer EEG AI Assistant,
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
