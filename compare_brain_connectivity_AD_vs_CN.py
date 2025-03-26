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
import mne
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def load_segment(path, label):
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose='ERROR')
    raw.filter(0.5, 45)
    raw.resample(128)
    data = raw.get_data(picks=raw.ch_names[:19])
    seg = data[:, :512]  # 4s
    return seg, raw.ch_names[:19], label

def compute_corr_graph(seg, ch_names, threshold=0.5):
    corr = np.corrcoef(seg)
    np.fill_diagonal(corr, 0)

    # Graph
    G = nx.Graph()
    for ch in ch_names:
        G.add_node(ch)
    for i in range(19):
        for j in range(i + 1, 19):
            if abs(corr[i, j]) >= threshold:
                G.add_edge(ch_names[i], ch_names[j], weight=corr[i, j])
    return corr, G

# === Paramètres ===
AD_PATH = "/workspace/memory_os_ai/alz/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
CN_PATH = "/workspace/memory_os_ai/alz/sub-002/eeg/sub-002_task-eyesclosed_eeg.set"
SAVE_PATH = "/workspace/memory_os_ai/alz/connectivity_comparison.png"

seg_ad, names, _ = load_segment(AD_PATH, "AD")
seg_cn, _, _ = load_segment(CN_PATH, "CN")

corr_ad, G_ad = compute_corr_graph(seg_ad, names)
corr_cn, G_cn = compute_corr_graph(seg_cn, names)

# === VISU 2x2 ===
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Figure 2A CN
axs[0, 0].imshow(corr_cn, cmap='coolwarm', vmin=-1, vmax=1)
axs[0, 0].set_title("Matrice de corrélation EEG — CN")
axs[0, 0].set_xticks(np.arange(19))
axs[0, 0].set_xticklabels(names, rotation=90)
axs[0, 0].set_yticks(np.arange(19))
axs[0, 0].set_yticklabels(names)

# Figure 2B CN
nx.draw(G_cn, pos=nx.circular_layout(G_cn), ax=axs[0,1],
        with_labels=True, node_color='lightgreen',
        edge_color='gray', width=2)
axs[0,1].set_title("Graph connectivité — CN")

# Figure 2A AD
axs[1, 0].imshow(corr_ad, cmap='coolwarm', vmin=-1, vmax=1)
axs[1, 0].set_title("Matrice de corrélation EEG — AD")
axs[1, 0].set_xticks(np.arange(19))
axs[1, 0].set_xticklabels(names, rotation=90)
axs[1, 0].set_yticks(np.arange(19))
axs[1, 0].set_yticklabels(names)

# Figure 2B AD
nx.draw(G_ad, pos=nx.circular_layout(G_ad), ax=axs[1,1],
        with_labels=True, node_color='salmon',
        edge_color='gray', width=2)
axs[1,1].set_title("Graph connectivité — AD")

plt.tight_layout()
plt.savefig(SAVE_PATH)
print(f"✅ Figure 2 complète sauvegardée : {SAVE_PATH}")

# === Export publication (SVG vectoriel haute résolution)
svg_path = SAVE_PATH.replace(".png", ".svg")
plt.savefig(svg_path, format='svg')
print(f"✅ Version publication SVG sauvegardée : {svg_path}")

 # ------------------------------------------------------------------------------
# 📄 LICENCE - Creative Commons Attribution-NonCommercial-ShareAlike 4.0
#
# Ce script "adformer_hybrid_voting_full.py" fait partie du projet Alzheimer EEG AI Assistant,
# développé par Kocupyr Romain (rkocupyr@gmail.com).
#
# Vous êtes libres de :
# ✅ Partager — copier le script
# ✅ Adapter — le modifier et l’intégrer dans un autre projet
#
# Sous les conditions suivantes :
# 📌 Attribution — Vous devez mentionner l’auteur original (Kocupyr Romain)
# 📌 Non Commercial — Interdiction d’usage commercial sans autorisation
# 📌 Partage identique — Toute version modifiée doit être publiée sous la même licence
#
# 🔗 Licence complète : https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------
