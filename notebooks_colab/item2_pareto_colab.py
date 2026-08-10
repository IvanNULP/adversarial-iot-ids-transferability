# =============================================================
# ITEM 2 — Pareto-Optimality Formalization of AARDF's Threshold
# Selection (bi-objective: aggregate ASR trade-off vs. hidden-risk
# detection). Reuses the already-computed 25-combination sensitivity
# grid (Figure 17) — no new experiments, pure analysis.
# Copy each CELL separately into Google Colab.
# =============================================================

# ── CELL 0: Mount Drive + load the existing sensitivity grid ──
from google.colab import drive
drive.mount('/content/drive')

import os, json
import pandas as pd
import numpy as np

BASE = '/content/drive/MyDrive/adversarial_iot_paper'
S8 = f'{BASE}/results/section8_aardf_sata'
OUT_DIR = f'{BASE}/results/section10_item2_pareto'
os.makedirs(OUT_DIR, exist_ok=True)

with open(f'{S8}/aardf_sensitivity_v2.json') as f:
    summary = json.load(f)

df = pd.DataFrame(summary)
print(f"Loaded {len(df)} threshold configurations.")
print(df.to_string(index=False))


# ── CELL 1: Define the two competing objectives (both to be minimized) ──
df['ASR_gap'] = df['aardf_mean_ASR'] - df['naive_mean_ASR']          # Objective 1: aggregate-risk cost of AARDF vs naive
df['risk_exposure'] = 1 - df['naive_winner_flagged_frac']            # Objective 2: fraction of hidden surrogate-risk NOT caught

print(df[['RS_LOW', 'TR_HIGH', 'ASR_gap', 'risk_exposure']].to_string(index=False))


# ── CELL 2: Extract the Pareto-optimal (non-dominated) set ──
def is_dominated(row, others):
    # row is dominated if some other row is <= on both objectives
    # and strictly < on at least one.
    for _, o in others.iterrows():
        if (o['ASR_gap'] <= row['ASR_gap'] and o['risk_exposure'] <= row['risk_exposure']
                and (o['ASR_gap'] < row['ASR_gap'] or o['risk_exposure'] < row['risk_exposure'])):
            return True
    return False

df['pareto_optimal'] = ~df.apply(lambda r: is_dominated(r, df.drop(r.name)), axis=1)

pareto_set = df[df['pareto_optimal']].sort_values('ASR_gap')
print(f"\n{df['pareto_optimal'].sum()} of {len(df)} configurations are Pareto-optimal (non-dominated):\n")
print(pareto_set[['RS_LOW', 'TR_HIGH', 'ASR_gap', 'risk_exposure']].to_string(index=False))

# Is the primary configuration used in Table 7/8 (RS_low=0.45, TR_high=1.5) on the frontier?
primary = df[(df['RS_LOW'] == 0.45) & (df['TR_HIGH'] == 1.5)]
print("\nPrimary configuration (RS_low=0.45, TR_high=1.5):")
print(primary[['RS_LOW', 'TR_HIGH', 'ASR_gap', 'risk_exposure', 'pareto_optimal']].to_string(index=False))


# ── CELL 3: Save results ──
out = {
    'objectives': {
        'objective_1': 'ASR_gap = aardf_mean_ASR - naive_mean_ASR (minimize)',
        'objective_2': 'risk_exposure = 1 - naive_winner_flagged_frac (minimize)',
    },
    'grid': df.to_dict(orient='records'),
    'pareto_optimal_set': pareto_set.to_dict(orient='records'),
    'primary_config_is_pareto_optimal': bool(primary['pareto_optimal'].iloc[0]) if not primary.empty else None,
}
out_path = f'{OUT_DIR}/item2_pareto_results.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print('\nSaved ->', out_path)
