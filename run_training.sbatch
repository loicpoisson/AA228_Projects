#!/bin/bash
#SBATCH --job-name=lunar_dqn      # Nom du job
#SBATCH --output=logs/output_%j.txt    # Fichier de sortie (logs)
#SBATCH --error=logs/error_%j.txt      # Fichier d'erreur
#SBATCH --time=02:00:00           # Temps max (2 heures - ajuster si besoin)
#SBATCH --mem=4G                  # Mémoire RAM (suffisant pour ce type de DQN)
#SBATCH --partition=normal        # Partition de calcul

# --- Configuration et Exécution ---

# 1. Charger Python (Version stable utilisée précédemment)
module load python/3.10.13

# 2. Activer l'environnement virtuel
# Assurez-vous que le dossier 'venv' a été créé correctement
source venv/bin/activate

# 3. Lancer le script principal
# Le code s'exécute en mode "non interactif"
python main.py