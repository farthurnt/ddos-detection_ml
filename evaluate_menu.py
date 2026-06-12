#!/usr/bin/env python3
"""
Menu d'évaluation multiclasse (Random Forest)
Possibilité de régler le seuil de classification Normal/Attaque.
"""

import os
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# CHARGEMENT DES FICHIERS (dossier courant)
# ------------------------------------------------------------
def load_assets():
    assets = {}
    try:
        assets['rf'] = joblib.load("random_forest_model.joblib")
        print("Random Forest chargé.")
    except Exception as e:
        print(f"Erreur chargement Random Forest : {e}")
        exit(1)

    assets['scaler'] = joblib.load("scaler.joblib")
    assets['label_encoder'] = joblib.load("label_encoder.joblib")
    assets['train_cols'] = pd.read_csv("training_columns.csv")['0'].tolist()
    print(f"Scaler, label encoder et {len(assets['train_cols'])} colonnes chargés.")
    return assets

# ------------------------------------------------------------
# NETTOYAGE ET PRÉPARATION (ajout automatique des colonnes manquantes)
# ------------------------------------------------------------
def prepare_data(df, train_cols):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    missing = [c for c in train_cols if c not in df.columns]
    if missing:
        print(f"Ajout de {len(missing)} colonne(s) manquante(s) avec des 0 : {missing[:5]}...")
        for col in missing:
            df[col] = 0
    return df[train_cols]

# ------------------------------------------------------------
# ANALYSE RANDOM FOREST (avec seuil ajustable)
# ------------------------------------------------------------
def analyze_rf(df, assets, normal_threshold=0.5):
    """
    normal_threshold : probabilité minimale pour rester classé 'Normal'.
    Si proba(Normal) < normal_threshold, on bascule vers la meilleure classe d'attaque.
    """
    le = assets['label_encoder']
    scaler = assets['scaler']
    rf = assets['rf']

    X = prepare_data(df, assets['train_cols'])
    if X is None:
        return

    X_scaled = scaler.transform(X)

    # Prédictions brutes
    preds_raw = rf.predict(X_scaled)
    proba_all = rf.predict_proba(X_scaled)
    # Indice de la classe 'Normal'
    normal_idx = list(le.classes_).index('Normal')
    proba_normal = proba_all[:, normal_idx]

    # Application du seuil
    adjusted_preds = []
    adjusted_proba = []
    for i in range(len(preds_raw)):
        if le.inverse_transform([preds_raw[i]])[0] == 'Normal' and proba_normal[i] < normal_threshold:
            # On choisit la classe non‑Normal avec la plus forte probabilité
            # Mettre la probabilité de Normal à 0 pour l'ignorer
            temp_proba = proba_all[i].copy()
            temp_proba[normal_idx] = 0
            new_pred = np.argmax(temp_proba)
            adjusted_preds.append(new_pred)
            adjusted_proba.append(temp_proba[new_pred])
        else:
            adjusted_preds.append(preds_raw[i])
            adjusted_proba.append(proba_all[i].max())
    adjusted_preds = np.array(adjusted_preds)
    max_proba = np.array(adjusted_proba)
    predicted_labels = le.inverse_transform(adjusted_preds)

    # Indicateurs
    normal_mask = predicted_labels == 'Normal'
    nb_normaux = normal_mask.sum()
    nb_anomalies = len(adjusted_preds) - nb_normaux
    taux = (nb_anomalies / len(adjusted_preds)) * 100

    print(f"\nSeuil Normal utilisé : {normal_threshold}")
    print("-" * 50)
    print("RÉSULTATS DE L'ANALYSE (Random Forest)")
    print("-" * 50)
    print(f"Flux normaux détectés : {nb_normaux}")
    print(f"Anomalies détectées   : {nb_anomalies}")
    print(f"Taux d'anomalies      : {taux:.2f} %")

    # Répartition par type
    print("\nRépartition des prédictions :")
    from collections import Counter
    for classe, count in Counter(predicted_labels).most_common():
        print(f"  {classe}: {count}")

    # Top 5 anomalies (hors Normal)
    anomaly_idx = np.where(~normal_mask)[0]
    if len(anomaly_idx) > 0:
        sorted_idx = anomaly_idx[np.argsort(max_proba[anomaly_idx])[::-1]]
        top5_idx = sorted_idx[:5]
        print("\nTop 5 anomalies (plus forte confiance) :")
        for i, idx in enumerate(top5_idx):
            print(f"  {i+1}. Type : {predicted_labels[idx]}, Confiance : {max_proba[idx]:.4f}")
    else:
        print("\nAucune anomalie détectée.")

    # Sauvegarde
    df['Prediction'] = predicted_labels
    df['Confiance'] = max_proba
    output = "resultat_analyse.csv"
    df.to_csv(output, index=False)
    print(f"\nRésultat complet sauvegardé dans {output}")

# ------------------------------------------------------------
# MENU PRINCIPAL
# ------------------------------------------------------------
def main():
    print("\n" + "-" * 50)
    print("  DÉTECTEUR DE TYPES DE DDoS (Random Forest) par Expedy & Arthur")
    print("\n" + "-" * 50)
    assets = load_assets()
    current_df = None
    current_threshold = 0.5   # valeur par défaut

    while True:
        print("\n--- MENU ---")
        print("1. Charger un fichier CSV")
        print("2. Analyser avec Random Forest")
        print("3. Changer le seuil de Normal (actuel : {:.2f})".format(current_threshold))
        print("4. Afficher les colonnes du fichier chargé")
        print("5. Vérifier la compatibilité des colonnes")
        print("0. Quitter")
        choix = input("Votre choix : ").strip()

        if choix == '1':
            path = input("Chemin complet du fichier CSV : ").strip()
            if not os.path.exists(path):
                print("Fichier introuvable.")
                continue
            try:
                current_df = pd.read_csv(path)
                print(f"Fichier chargé : {os.path.basename(path)} ({current_df.shape[0]} lignes, {current_df.shape[1]} colonnes)")
            except Exception as e:
                print(f"Erreur de lecture : {e}")

        elif choix == '2':
            if current_df is None:
                print("Chargez d'abord un fichier (option 1).")
            else:
                analyze_rf(current_df.copy(), assets, normal_threshold=current_threshold)

        elif choix == '3':
            try:
                new_thresh = float(input("Nouveau seuil (entre 0 et 1) : ").strip())
                if 0 <= new_thresh <= 1:
                    current_threshold = new_thresh
                    print(f"Seuil mis à {current_threshold:.2f}")
                else:
                    print("Valeur hors intervalle [0,1].")
            except ValueError:
                print("Nombre invalide.")

        elif choix == '4':
            if current_df is not None:
                print("Colonnes du fichier :")
                print(current_df.columns.tolist())
            else:
                print("Aucun fichier chargé.")

        elif choix == '5':
            if current_df is not None:
                missing = [c for c in assets['train_cols'] if c not in current_df.columns]
                if missing:
                    print(f"Colonnes manquantes ({len(missing)}) : {missing[:10]}...")
                else:
                    print("Toutes les colonnes sont présentes.")
            else:
                print("Aucun fichier chargé.")

        elif choix == '0':
            print("Au revoir !")
            break
        else:
            print("Option invalide.")

if __name__ == "__main__":
    main()
