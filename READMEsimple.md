
# Alz Simple – Détection d’Alzheimer via EEG simulé

*Alz Simple* est une version simplifiée d’un assistant IA pour détecter les stades de la maladie d’Alzheimer ("Sain", "Début", "Modéré", "Avancé") à partir de signaux EEG simulés. Ce projet fait partie de l’initiative *Alzheimer EEG AI Assistant*.

**Auteur** : Romain Kocupyr  
**Email** : rkocupyr@gmail.com  
**Licence** : Creative Commons BY-NC-SA 4.0 ([voir détails](https://creativecommons.org/licenses/by-nc-sa/4.0/))  

---

## Objectif
Créer un modèle IA capable de :  
- Générer des signaux EEG simulés réalistes (ARIMA + bruit).  
- Classifier les stades d’Alzheimer avec une précision élevée.  
- Proposer des recommandations de stimulation VR personnalisées.  

---

## Fonctionnalités
- **Simulation EEG** : Génération de 10 000 échantillons (512 Hz, 19 électrodes) avec bruit réaliste (gaussien, artefacts 50 Hz, clignements exponentiels).  
- **Prétraitement** : Filtre de Kalman pour réduire le bruit.  
- **Modèle IA** : CNN-LSTM (TensorFlow) avec apprentissage incrémental via générateur.  
- **Résultats** : Précision simulée de 100 % sur 4 classes, robustesse au bruit.  
- **Sorties** : Matrice de confusion, recommandations VR (ex. : "15 min Theta" pour "Modéré").  
- **Portabilité** : Exécution automatique CPU/GPU.  

---

## Prérequis
- Python 3.8+  
- Dépendances :  
  ```
  pip install -r requirements.txt
  ```

Voir requirements.txt pour la liste complète (NumPy, TensorFlow, etc.).  

Utilisation

Lancer le script :  
```
alz_simple.py
```
