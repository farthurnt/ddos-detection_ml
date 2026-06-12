# DDoS Detection by Machine Learning

Système de détection et classification d'attaques DDoS en temps réel, basé sur un modèle Random Forest multiclasse entraîné sur le dataset CIC-IDS 2019.

---

## Objectif

Détecter et classifier automatiquement 7 types d'attaques DDoS dans un flux réseau réel, en intégrant les alertes dans un SIEM (Wazuh) via Suricata.

---

## Architecture du pipeline

```
Trafic réseau → Suricata → Script Python → Wazuh (syslog) → Dashboard Kibana
```

---

## Types d'attaques détectés

| Label original | Classe détectée |
|---|---|
| BENIGN | Normal |
| DrDoS_DNS | DNS Amplification |
| DrDoS_NTP | NTP Amplification |
| Syn | SYN Flood |
| UDP | UDP Flood |
| ICMP | ICMP Flood |
| WebDDoS | HTTP Flood |
| Slowloris | Slowloris |

---

## Dataset

- **Source :** [CIC-IDS 2019](https://www.unb.ca/cic/datasets/ddos-2019.html) — University of New Brunswick
- **Volume :** ~13 Go de trafic réseau réel
- **Échantillonnage :** 2% par chunk de 50 000 lignes (stratifié)

> Le dataset n'est pas inclus dans ce dépôt en raison de sa taille. Téléchargez-le depuis le lien ci-dessus et placez les fichiers CSV dans le dossier `data_2019/`.

---

## Modèles entraînés

| Fichier | Description |
|---|---|
| `random_forest_model.joblib` | Classificateur multiclasse principal |
| `isolation_forest_model.joblib` | Détection d'anomalies non supervisée |
| `scaler.joblib` | StandardScaler pour la normalisation |
| `label_encoder.joblib` | Encodage des labels de classes |
| `training_columns.csv` | Liste des features utilisées à l'entraînement |

---

## Structure du projet

```
ddos-detection-ml/
├── train_multiclass.py       # Entraînement Random Forest + Isolation Forest
├── evaluate_menu.py          # Évaluation interactive du modèle
├── soc_integration.py        # Intégration pipeline SOC (Suricata → Wazuh)
├── training_columns.csv      # Features sélectionnées
├── resultat_analyse.csv      # Résultats d'évaluation
├── *.joblib                  # Modèles et encodeurs sauvegardés
└── data_2019/                # Dataset CIC-IDS 2019 (non inclus)
```

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/farthurnt/ddos-detection_ml.git
cd ddos-detection_ml

# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install pandas numpy scikit-learn joblib
```

---

## Utilisation

### 1. Entraîner le modèle
```bash
python3 train_multiclass.py
```
Génère les fichiers `.joblib` et `training_columns.csv`.

### 2. Évaluer le modèle
```bash
python3 evaluate_menu.py
```
Menu interactif pour tester et analyser les performances.

### 3. Intégration SOC
```bash
python3 soc_integration.py
```
Lance le pipeline temps réel : Suricata → classification → envoi syslog vers Wazuh.

---

## Paramètres clés

| Paramètre | Valeur |
|---|---|
| `N_ESTIMATORS` | 50 arbres |
| `SAMPLE_FRAC` | 2% par chunk |
| `CHUNK_SIZE` | 50 000 lignes |
| `TEST_SIZE` | 20% |
| `class_weight` | balanced |

---

## Environnement de test

- OS : Kali Linux (VirtualBox)
- Attaques simulées : SYN Flood, UDP Flood, ICMP Flood, Slowloris
- SIEM : Wazuh avec règles personnalisées
- IDS : Suricata

---

## Auteur

**NDOUNG TIPANE Franck Arthur**  
Analyste SSI | Administrateur Réseau & Sécurité  
[LinkedIn](https://www.linkedin.com/in/franckarthur-ndoung-tipane-a00638173/) • [Portfolio](https://franckcyber.page.gd)
