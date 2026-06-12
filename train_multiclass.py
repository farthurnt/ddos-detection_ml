#!/usr/bin/env python3
"""
Entrainement multiclasse CIC-IDS 2019 (données dans data_2019/)
Corrige espaces dans colonnes, filtre colonnes non numériques.
Production : random_forest_model.joblib, isolation_forest_model.joblib,
             scaler.joblib, label_encoder.joblib, training_columns.csv
"""

import os, glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ----------------- CONFIG -----------------
DATA_DIR = "data_2019"
CHUNK_SIZE = 50000
SAMPLE_FRAC = 0.02
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_JOBS = 2
N_ESTIMATORS = 50

LABEL_MAPPING = {
    'BENIGN': 'Normal',
    'DrDoS_DNS': 'DNS Amplification',
    'DrDoS_NTP': 'NTP Amplification',
    'Syn': 'SYN Flood',
    'UDP': 'UDP Flood',
    'ICMP': 'ICMP Flood',
    'WebDDoS': 'HTTP Flood',
    'Slowloris': 'Slowloris',
}

# ----------------- 1. CHARGEMENT ÉCHANTILLONNÉ -----------------
print("Recherche des CSV dans", DATA_DIR)
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))
print(f"{len(csv_files)} fichiers trouvés.")

chunks_list = []

for file in csv_files:
    print(f"Traitement de {os.path.basename(file)}...")
    try:
        chunk_iter = pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False)
        for chunk in chunk_iter:
            if len(chunk) == 0:
                continue
            # Nettoyage des noms de colonnes (espaces)
            chunk.columns = chunk.columns.str.strip()

            # Échantillonnage
            sample = chunk.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)

            # Mapping des labels
            sample['Label'] = sample['Label'].map(LABEL_MAPPING).fillna('Autre')

            # Suppression infinis / NaN
            sample.replace([np.inf, -np.inf], np.nan, inplace=True)
            sample.dropna(inplace=True)

            if len(sample) > 0:
                chunks_list.append(sample)
    except Exception as e:
        print(f"    Erreur sur {file}: {e}")

if not chunks_list:
    print("Aucune donnée chargée.")
    exit(1)

df_total = pd.concat(chunks_list, ignore_index=True)
print(f"\nTaille échantillon : {df_total.shape[0]:,} lignes")
print(df_total['Label'].value_counts())

df_total.drop_duplicates(inplace=True)

# ----------------- 2. ENCODAGE LABELS -----------------
le = LabelEncoder()
y = le.fit_transform(df_total['Label'])
joblib.dump(le, "label_encoder.joblib")

# ----------------- 3. SÉLECTION DES COLONNES NUMÉRIQUES -----------------
# On retire la colonne Label et on ne garde que les colonnes numériques
X_all = df_total.drop(columns=['Label'])
# Colonnes non numériques à exclure explicitement (au cas où)
non_numeric = ['Unnamed: 0', 'Flow ID', ' Source IP', ' Destination IP', ' Timestamp',
               'SimillarHTTP', ' Inbound']  # on ajuste après strip
# On applique strip à toutes les colonnes de X_all déjà fait, mais on vérifie
numeric_cols = []
for col in X_all.columns:
    col_stripped = col.strip()
    if X_all[col].dtype in [np.int64, np.float64, np.float32, np.int32, np.bool_]:
        numeric_cols.append(col)
    # Par sécurité, on peut aussi tester si elle n'est pas dans non_numeric
    # mais le dtype suffit normalement.

X = X_all[numeric_cols]
training_cols = X.columns.tolist()
pd.Series(training_cols).to_csv("training_columns.csv", index=False)

print(f"\nColonnes utilisées ({len(training_cols)}) :")
print(training_cols[:10], "..." if len(training_cols) > 10 else "")

# ----------------- 4. TRAIN/TEST -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]:,} lignes, Test: {X_test.shape[0]:,} lignes")

# ----------------- 5. SCALER -----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.joblib")

# ----------------- 6. RANDOM FOREST -----------------
print("\nEntrainement Random Forest...")
rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)
rf.fit(X_train_scaled, y_train)
joblib.dump(rf, "random_forest_model.joblib")

from sklearn.metrics import classification_report
y_pred_rf = rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# ----------------- 7. ISOLATION FOREST -----------------
print("\nEntrainement Isolation Forest...")
normal_idx = le.transform(['Normal'])[0]
contam = (y_train != normal_idx).mean()
iso = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=contam,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)
iso.fit(X_train_scaled)
joblib.dump(iso, "isolation_forest_model.joblib")

print("\n=== Terminé ===")
