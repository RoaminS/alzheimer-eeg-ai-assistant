"""
plot_brain_connectivity.py.py – Affiché image Matrice et Positionnement Circulaire

Auteur : Kocupyr Romain créateur et chef de projet/ Dev= multi_gpt_api, grok3
Licence : Creative Commons BY-NC-SA 4.0
https://creativecommons.org/licenses/by-nc-sa/4.0/
""" 

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# === Config
EEG_PATH = "/workspace/memory_os_ai/alz/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"  # Exemple AD
LABEL = "AD"  # ou "CN" ou "Sain"
SAVE_DIR = "/workspace/memory_os_ai/alz/"

# === Chargement EEG
raw = mne.io.read_raw_eeglab(EEG_PATH, preload=True, verbose='ERROR')
raw.filter(0.5, 45)
raw.resample(128)
data = raw.get_data(picks=raw.ch_names[:19])  # shape : (19, N)

# === Segment 4s
seg = data[:, :512]  # shape : (19, 512)

# === Matrice de corrélation
corr = np.corrcoef(seg)
np.fill_diagonal(corr, 0)

# === Plot Matrice (Figure 2A)
plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.title(f"Matrice de corrélation EEG — {LABEL}")
plt.colorbar(label="Corrélation inter-canaux")
plt.xticks(ticks=np.arange(19), labels=raw.ch_names[:19], rotation=90)
plt.yticks(ticks=np.arange(19), labels=raw.ch_names[:19])
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, f"connectivity_corr_matrix_{LABEL}.png"))
print(f"✅ Figure 2A sauvegardée : connectivity_corr_matrix_{LABEL}.png")

# === Graph (Figure 2B)
G = nx.Graph()
channels = raw.ch_names[:19]

# Ajout des noeuds
for ch in channels:
    G.add_node(ch)

# Seuil pour lien fort
threshold = 0.5
for i in range(19):
    for j in range(i + 1, 19):
        if abs(corr[i, j]) >= threshold:
            G.add_edge(channels[i], channels[j], weight=corr[i, j])

# Positionnement circulaire
pos = nx.circular_layout(G)
plt.figure(figsize=(6, 6))
edges = G.edges(data=True)
weights = [abs(d["weight"]) * 2 for (_, _, d) in edges]
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', width=weights)
plt.title(f"Connectivité cérébrale (corr ≥ {threshold}) — {LABEL}")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, f"connectivity_graph_{LABEL}.png"))
print(f"✅ Figure 2B sauvegardée : connectivity_graph_{LABEL}.png")





# ------------------------------------------------------------------------------
# 📄 LICENCE - Creative Commons Attribution-NonCommercial-ShareAlike 4.0
#
# Ce script "plot_brain_connectivity.py" fait partie du projet Alzheimer EEG AI Assistant,
# développé par Kocupyr Romain (rkocupyr@gmail.com).
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
