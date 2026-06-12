#!/usr/bin/env python3
"""
Integration SOC : Suricata -> Random Forest -> Wazuh (syslog)
Surveille /var/log/suricata/eve.json et alerte Wazuh en cas d'attaque DDoS.
"""

import json
import time
import os
import pandas as pd
import numpy as np
import joblib
import logging
import logging.handlers

# ----------------- CONFIGURATION -----------------
EVE_LOG = "/var/log/suricata/eve.json"
MODEL_PATH = "/opt/soc_model/random_forest_model.joblib"
SCALER_PATH = "/opt/soc_model/scaler.joblib"
LABEL_ENCODER_PATH = "/opt/soc_model/label_encoder.joblib"
COLUMNS_PATH = "/opt/soc_model/training_columns.csv"
NORMAL_THRESHOLD = 0.9

# Chargement des assets
rf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
train_cols = pd.read_csv(COLUMNS_PATH)['0'].tolist()
normal_idx = list(le.classes_).index('Normal')

# Configuration syslog
logger = logging.getLogger('DDoS_Detector')
logger.setLevel(logging.ALERT)
handler = logging.handlers.SysLogHandler(address=('127.0.0.1', 514))
logger.addHandler(handler)

# ----------------- FONCTIONS -----------------
def event_to_features(event):
    feats = {col: 0.0 for col in train_cols}
    if 'flow' not in event:
        return pd.DataFrame([feats])[train_cols]
    flow = event['flow']
    feats['Flow Duration'] = flow.get('flow_duration', 0)
    feats['Total Fwd Packets'] = flow.get('pkts_toserver', 0)
    feats['Total Backward Packets'] = flow.get('pkts_toclient', 0)
    feats['Total Length of Fwd Packets'] = flow.get('bytes_toserver', 0)
    feats['Total Length of Bwd Packets'] = flow.get('bytes_toclient', 0)
    feats['Source Port'] = flow.get('src_port', 0)
    feats['Destination Port'] = flow.get('dest_port', 0)
    return pd.DataFrame([feats])[train_cols]

def process_event(event):
    if event.get('event_type') != 'flow':
        return
    src_ip = event.get('src_ip', '?')
    dest_ip = event.get('dest_ip', '?')
    X = event_to_features(event)
    X_scaled = scaler.transform(X)
    proba = rf.predict_proba(X_scaled)[0]
    proba_normal = proba[normal_idx]
    pred_raw = rf.predict(X_scaled)[0]

    if le.inverse_transform([pred_raw])[0] == 'Normal' and proba_normal < NORMAL_THRESHOLD:
        proba_temp = proba.copy()
        proba_temp[normal_idx] = 0
        pred = np.argmax(proba_temp)
        confiance = proba_temp[pred]
    else:
        pred = pred_raw
        confiance = proba.max()
    classe = le.inverse_transform([pred])[0]

    if classe != 'Normal':
        msg = f"DDoS Alert: Type={classe}, Confidence={confiance:.2f}, Src={src_ip}, Dst={dest_ip}"
        print(f"[ALERTE] {msg}")
        logger.alert(msg)

# ----------------- BOUCLE PRINCIPALE -----------------
print("Surveillance de Suricata eve.json...")
print(f"Seuil Normal : {NORMAL_THRESHOLD}")

with open(EVE_LOG, 'r') as f:
    f.seek(0, os.SEEK_END)
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.1)
            continue
        try:
            event = json.loads(line)
            process_event(event)
        except (json.JSONDecodeError, KeyError):
            pass
