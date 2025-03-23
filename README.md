# Alzheimer-eeg-ai-assistant

# 🧠 Alzheimer EEG AI Assistant

Assistant Cognitif IA temps réel pour la détection et le soutien des troubles neurodégénératifs via signaux EEG.  
Un projet de recherche autonome, médicalement orienté, utilisant des technologies IA avancées et un pipeline EEG complet.

---

## 🚀 Objectif

 Créer une IA adaptative capable de :

 - Lire les signaux cérébraux EEG
 - Détecter les stades d’Alzheimer avec précision
 - Simuler l'activité cérébrale pour entraîner des modèles
 - Offrir des recommandations de stimulation cognitive personnalisées (VR Alpha, Theta, Delta)

---

## 🧠 Fonctionnalités clés

- ✅ Nettoyage dynamique des signaux EEG (NaN, artefacts, canaux plats)
- ✅ Filtrage UKF adaptatif bio-inspiré simulant les dynamiques cérébrales
- ✅ Extraction spectrale (STFT, bandes Delta/Theta/Alpha/Beta)
- ✅ Simulation EEG via ARIMA selon niveaux cognitifs
- ✅ Modèle IA LSTM bidirectionnel + Attention (TensorFlow)
- ✅ Classification multi-classes : Sain / Début / Modéré / Avancé
- ✅ Indexation FAISS GPU pour déduplication intelligente et équilibrage des classes
- ✅ Export ONNX + TFLite pour exécution sur terminal embarqué (Edge IA)

---

## 📈 Résultats obtenus

| Indicateur               | Résultat      |
|--------------------------|---------------|
| Précision par patient    | > 90%         |
| Sensibilité Alzheimer    | > 95%         |
| Cas incertains détectés  | Oui (proba < 0.5) |
| Export                   | ONNX / TFLite |


📊 Recommandations personnalisées :
- `Sain` → Aucune stimulation
- `Début` → VR Alpha 10 min
- `Modéré` → VR Theta + relaxation 15 min
- `Avancé` → VR Delta + surveillance médicale urgente



## 📦 Structure du dépôt
```
alzheimer-eeg-ai-assistant/
├── README.md ✅
├── LICENSE ✅ (CC BY-NC-SA 4.0)
├── model/
│   ├── alz_model.keras
│   ├── alz_model.onnx
│   └── alz_model.tflite
├── data/
│   ├── participants.tsv
│   ├── eeg_data.pkl
│   ├── faiss_index.bin
│   └── faiss_metadata.pkl
├── src/
│   ├── preprocess.py
│   ├── ukf_filter.py
│   ├── train_model.py
│   ├── inference.py
│   └── simulate_data.py
├── assets/
│   ├── alz_confusion_matrix_patient.png
│   ├── probability_histogram.png
│   └── flat_channel_barplot.png
├── requirements.txt ✅
└── .gitignore ✅
```



## 🛠 Installation rapide

```
git clone https://github.com/votre-profil/alzheimer-eeg-ai-assistant.git
cd alzheimer-eeg-ai-assistant
pip install -r requirements.txt
```

---

🔗 Exports
🧠 Modèle IA disponible au format :

.keras (entraînement et test local)

.onnx (interopérabilité IA)

.tflite (exécution sur terminal embarqué – mobile, casque, etc.)

---


🤝 Collaboration
🎯 Je cherche à collaborer avec :

Chercheurs en neurosciences et cognition

Universités et hôpitaux ouverts à l’innovation

Développeurs open-source en IA/BCI

Acteurs de la santé numérique et cognitive

📩 Pour toute proposition ou idée → romainsantoli@gmail.com

📄 Licence


---

Ce projet est publié sous la licence Creative Commons BY-NC-SA 4.0.


Vous êtes libre de :

✅ Partager — copier et redistribuer le matériel

✅ Adapter — remixer, transformer, et créer à partir du matériel


Sous conditions :

📌 Attribution — mentionner l’auteur original (Kocupyr Romain)

📌 Non Commercial — pas d’usage commercial sans accord explicite

📌 Partage identique — redistribuer les versions modifiées sous la même licence


🔗 Lire la licence complète


---
✨ Crédits
Ce projet a été développé en autonomie complète par Kocupyr Romain, avec le soutien des IA : Chatgpt & Grok.

🎙️ "Je ne veux pas remplacer la médecine. Je veux l'améliorer et la rendre accessible au plus grand nombre." — K.R.
