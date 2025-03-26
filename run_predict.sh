"""
-Script interactif
-Demande à l’utilisateur le chemin vers un fichier .set
-Lance run_predict.py automatiquement avec la bonne valeur
-Gère les erreurs (fichier manquant / mauvais format)

Le rendre executable: 
chmod +x /chemin/dossier/run_predict.sh

Puis pour l'executer: (être dans le bon dossier :) )
./run_predict.sh

Licence : Creative Commons BY-NC-SA 4.0
Auteurs : Kocupyr Romain créateur et chef de projet, dev = GPT multi_gpt_api (OpenAI), Grok3, kocupyr romain
"""

#!/bin/bash

echo "🧠 === Alzheimer EEG Prédicteur ==="
echo ""
read -p "Entrez le chemin complet vers le fichier EEG (.set) : " EEG_PATH

if [ ! -f "$EEG_PATH" ]; then
  echo "❌ Fichier introuvable : $EEG_PATH"
  exit 1
fi

echo "✅ Fichier trouvé : $EEG_PATH"
echo ""

# Modifier dynamiquement le script Python
TEMP_SCRIPT="run_predict_temp.py"
cp run_predict.py $TEMP_SCRIPT
sed -i "s|EEG_PATH = .*|EEG_PATH = \"$EEG_PATH\"|" $TEMP_SCRIPT

echo "🚀 Lancement de la prédiction..."
python $TEMP_SCRIPT

# Nettoyage
rm $TEMP_SCRIPT


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

