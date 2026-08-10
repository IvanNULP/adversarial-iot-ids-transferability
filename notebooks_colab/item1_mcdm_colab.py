# =============================================================
# ITEM 1 — Weighted-MCDM Baseline Comparison vs AARDF vs Naive
# Reuses exact aardf_decide() logic and thresholds from the
# Section 6.9 retrospective-validation notebook, so results stay
# consistent with already-published Table 7/8 and Figure 13/14.
# Copy each CELL separately into Google Colab.
# =============================================================

# ── CELL 0: Mount Drive + safety check ──
from google.colab import drive
drive.mount('/content/drive')

import os

BASE = '/content/drive/MyDrive/adversarial_iot_paper'
S6 = f'{BASE}/results/section6_v2'
S7 = f'{BASE}/results/section7_v2'
OUT_DIR = f'{BASE}/results/section9_item1_mcdm'
os.makedirs(OUT_DIR, exist_ok=True)

print('--- section6_v2 ---')
print(os.listdir(S6) if os.path.isdir(S6) else 'MISSING')
print('--- section7_v2 ---')
print(os.listdir(S7) if os.path.isdir(S7) else 'MISSING')


# ── CELL 1: Load result files (same paths/keys as Section 6.9) ──
import json, pandas as pd

table6 = pd.read_csv(f'{S6}/table6_v2.csv')

with open(f'{S7}/wb_results_v2.json') as f:
    wb = json.load(f)
with open(f'{S7}/ca_results_v2.json') as f:
    ca = json.load(f)
with open(f'{S7}/cd_results_v2.json') as f:
    cd = json.load(f)
with open(f'{S7}/stat_results_v2.json') as f:
    stat = json.load(f)
with open(f'{S7}/defense_results_v2.json') as f:
    defense = json.load(f)

datasets = ['CIC-IDS 2017', 'UNSW-NB15', 'Gotham IoT 2025', 'CIC-YNU-IoTMal 2026']
models = ['RF', 'XGBoost', 'MLP', 'CNN']

print('table6_v2 columns:', list(table6.columns))


# ── CELL 2: Extract per (dataset, model) Robustness Score ──
rs_block = stat['robustness_scores']

rs_values = {(d, m): [] for d in datasets for m in models}
for k, v in rs_block.items():
    d, m = v['ds'], v['model']
    if (d, m) in rs_values:
        rs_values[(d, m)].append(v['rs'])

rs_table = {key: (sum(vals) / len(vals) if vals else None) for key, vals in rs_values.items()}


# ── CELL 3: AARDF decision logic — EXACT reuse from Section 6.9 ──
RS_LOW = 0.45
TR_HIGH = 1.5
ASR_TRANSFER_HIGH = 0.45
ASR_REDUCTION_MIN = 0.30
ACC_LOSS_MAX = 0.20

def clean_acc(dataset, model):
    row = table6[table6['Dataset'] == dataset]
    if row.empty:
        return None
    col = f'{model}_Acc'
    return float(row.iloc[0][col]) if col in table6.columns else None

def whitebox_asr_mean(dataset, model, wb_json):
    vals = [v['asr'] for v in wb_json.values() if v['ds'] == dataset and v['model'] == model]
    return sum(vals) / len(vals) if vals else None

def tr_max_into(dataset, model, ca_json):
    vals = [v['tr'] for v in ca_json.values() if v['ds'] == dataset and v['tgt'] == model]
    return max(vals) if vals else 0.0

CD_MODEL_ALIAS = {'XGBoost': 'XGB'}  # cd_results_v2.json stores XGBoost as 'XGB';
                                       # RF and MLP keys match directly. CNN is
                                       # genuinely absent from cd_results_v2.json
                                       # by design (SATA cross-dataset alignment
                                       # covers only RF/XGB/MLP, 36 entries total).

def asr_cross_into(dataset, model, cd_json):
    # Only defined for CIC-IDS 2017 <-> UNSW-NB15 (no cross-dataset
    # counterpart exists for Gotham IoT 2025 / CIC-YNU-IoTMal 2026
    # in the current benchmark design), and only for RF/XGBoost/MLP
    # (CNN excluded from the SATA cross-dataset alignment protocol).
    if dataset not in ('CIC-IDS 2017', 'UNSW-NB15'):
        return None
    cd_model = CD_MODEL_ALIAS.get(model, model)
    other = 'UNSW-NB15' if dataset == 'CIC-IDS 2017' else 'CIC-IDS 2017'
    vals = [v['asr_tgt'] for v in cd_json.values()
            if v['src_ds'] == other and v['tgt_ds'] == dataset and v['model'] == cd_model]
    return max(vals) if vals else None

def fs_effect(dataset, model, defense_json):
    reductions, acc_losses = [], []
    for v in defense_json.values():
        if v['ds'] == dataset and v['model'] == model:
            if v['asr_before'] > 0:
                reductions.append((v['asr_before'] - v['asr_fs']) / v['asr_before'])
            if v['acc_clean']:
                acc_losses.append((v['acc_clean'] - v['clean_fs']) / v['acc_clean'])
    reduction = sum(reductions) / len(reductions) if reductions else None
    acc_loss = sum(acc_losses) / len(acc_losses) if acc_losses else None
    return reduction, acc_loss

def aardf_decide(dataset, model):
    rs = rs_table.get((dataset, model))
    if rs is None:
        return 'UNKNOWN', {}
    trace = {'RS': rs}
    if rs < RS_LOW:
        return 'REASSESS', trace
    tr_max = tr_max_into(dataset, model, ca)
    trace['TR_max'] = tr_max
    flags = []
    if tr_max > TR_HIGH:
        flags.append('DATASET_SPECIFIC_SURROGATE_RISK')
    asr_cross = asr_cross_into(dataset, model, cd)
    trace['ASR_cross'] = asr_cross
    if asr_cross is not None and asr_cross > ASR_TRANSFER_HIGH:
        flags.append('CROSS_GENERATION_RISK')
    trace['flags'] = flags
    if flags:
        reduction, acc_loss = fs_effect(dataset, model, defense)
        trace['FS_reduction'] = reduction
        trace['FS_acc_loss'] = acc_loss
        if reduction is not None and acc_loss is not None and reduction >= ASR_REDUCTION_MIN and acc_loss <= ACC_LOSS_MAX:
            return 'DEPLOY_WITH_DEFENSE(FS)', trace
        return 'REASSESS', trace
    return 'DEPLOY', trace

decisions = {}
for d in datasets:
    for m in models:
        decisions[(d, m)] = aardf_decide(d, m)

for k, v in decisions.items():
    print(k, '->', v[0])


# ── CELL 4: Build the 16 combos with RS, TR_max, ASR_cross ──
combos = []
for d in datasets:
    for m in models:
        rs = rs_table.get((d, m))
        tr_max = tr_max_into(d, m, ca)
        asr_cross = asr_cross_into(d, m, cd)          # None for Gotham / IoTMal
        asr_wb = whitebox_asr_mean(d, m, wb)           # defined for all 16
        combos.append({
            'dataset': d, 'model': m,
            'RS': rs, 'TR_max': tr_max,
            'ASR_cross': asr_cross, 'ASR_whitebox': asr_wb,
            'aardf_decision': decisions[(d, m)][0],
        })

combo_df = pd.DataFrame(combos)
print(combo_df.to_string(index=False))


# ── CELL 5: Weighted-MCDM baseline (Fessi et al. [50] / Youssef et al. [52] style) ──
# Weighted linear combination of normalized criteria.
# Weights below are a starting point consistent with AARDF's own
# emphasis on RS as the primary gate (RS_LOW check fires first in
# Algorithm 1) — confirm/adjust before use in the paper.
W_RS  = 0.4   # higher RS = better
W_TR  = 0.3   # higher TR_max = worse
W_ASR = 0.3   # higher ASR = worse
MCDM_THRESHOLD = 0.5   # score >= threshold -> DEPLOY, else REASSESS

import numpy as np

def normalize(series):
    arr = np.array(series, dtype=float)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if vmax == vmin:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)

def run_mcdm(asr_column):
    """asr_column: 'ASR_cross' (strict, n=6: RF/XGBoost/MLP x CIC/UNSW)
    or 'ASR_whitebox' (all 16)."""
    sub = combo_df.dropna(subset=['RS', 'TR_max', asr_column]).copy()
    rs_n = normalize(sub['RS'])
    tr_n = 1 - normalize(sub['TR_max'])
    asr_n = 1 - normalize(sub[asr_column])
    sub['mcdm_score'] = W_RS * rs_n + W_TR * tr_n + W_ASR * asr_n
    sub['mcdm_decision'] = np.where(sub['mcdm_score'] >= MCDM_THRESHOLD, 'DEPLOY', 'REASSESS')
    return sub

strict_df = run_mcdm('ASR_cross')      # n=6: RF/XGBoost/MLP x {CIC-IDS 2017, UNSW-NB15}
                                        # (CNN has no ASR_cross by design - excluded from
                                        # the SATA cross-dataset alignment protocol)
extended_df = run_mcdm('ASR_whitebox')  # all 16 combos

print(f"\n=== STRICT (ASR_cross, n={len(strict_df)}) ===")
print(strict_df[['dataset','model','mcdm_score','mcdm_decision','aardf_decision']].to_string(index=False))

print(f"\n=== EXTENDED (ASR_whitebox proxy, n={len(extended_df)}) ===")
print(extended_df[['dataset','model','mcdm_score','mcdm_decision','aardf_decision']].to_string(index=False))


# ── CELL 6: Decision agreement (AARDF vs weighted-MCDM), per-combo ──
def aardf_binary(label):
    # Collapse AARDF's 3-way output to DEPLOY / REASSESS for a fair
    # binary comparison against the MCDM baseline (DEPLOY_WITH_DEFENSE
    # counts as DEPLOY since the combo is ultimately fielded).
    return 'REASSESS' if label in ('REASSESS', 'UNKNOWN') else 'DEPLOY'

for label, df in [('STRICT', strict_df), ('EXTENDED', extended_df)]:
    df = df.copy()
    df['aardf_binary'] = df['aardf_decision'].apply(aardf_binary)
    agree = (df['aardf_binary'] == df['mcdm_decision']).mean()
    print(f"{label}: AARDF vs weighted-MCDM decision agreement = {agree:.2f} ({len(df)} combos)")


# ── CELL 7: Aggregate-ASR comparison — AARDF vs weighted-MCDM vs naive ──
# Mirrors the exact per-dataset "pick a winner" comparison style used
# in Section 6.9 (naive = highest clean accuracy), so this is directly
# comparable to the already-published 52% / 70% retrospective result.

def eligible_aardf_model(d):
    eligible = {m: rs_table[(d, m)] for m in models
                if decisions[(d, m)][0] not in ('REASSESS', 'UNKNOWN') and rs_table.get((d, m)) is not None}
    return max(eligible, key=eligible.get) if eligible else None

def mcdm_winner_model(d, mcdm_df):
    sub = mcdm_df[mcdm_df['dataset'] == d]
    if sub.empty:
        return None
    return sub.loc[sub['mcdm_score'].idxmax(), 'model']

rows = []
for d in datasets:
    accs = {m: clean_acc(d, m) for m in models}
    accs = {m: a for m, a in accs.items() if a is not None}
    naive_model = max(accs, key=accs.get) if accs else None

    aardf_model = eligible_aardf_model(d) or naive_model
    mcdm_model_ext = mcdm_winner_model(d, extended_df)
    mcdm_model_strict = mcdm_winner_model(d, strict_df) if d in ('CIC-IDS 2017', 'UNSW-NB15') else None

    rows.append({
        'dataset': d,
        'naive_model': naive_model,
        'naive_ASR': whitebox_asr_mean(d, naive_model, wb) if naive_model else None,
        'aardf_model': aardf_model,
        'aardf_ASR': whitebox_asr_mean(d, aardf_model, wb) if aardf_model else None,
        'mcdm_model_extended': mcdm_model_ext,
        'mcdm_ASR_extended': whitebox_asr_mean(d, mcdm_model_ext, wb) if mcdm_model_ext else None,
        'mcdm_model_strict': mcdm_model_strict,
        'mcdm_ASR_strict': whitebox_asr_mean(d, mcdm_model_strict, wb) if mcdm_model_strict else None,
    })

result_df = pd.DataFrame(rows)
print(result_df.to_string(index=False))

print("\n=== Mean aggregate ASR across datasets ===")
print("Naive :", result_df['naive_ASR'].mean())
print("AARDF :", result_df['aardf_ASR'].mean())
print("MCDM (extended, all 4 datasets):", result_df['mcdm_ASR_extended'].mean())
print("MCDM (strict, CIC+UNSW only)   :", result_df['mcdm_ASR_strict'].dropna().mean())


# ── CELL 8: Save everything to Google Drive (mandatory) ──
out = {
    'weights': {'RS': W_RS, 'TR_max': W_TR, 'ASR': W_ASR},
    'mcdm_threshold': MCDM_THRESHOLD,
    'aardf_thresholds': {'RS_LOW': RS_LOW, 'TR_HIGH': TR_HIGH,
                          'ASR_TRANSFER_HIGH': ASR_TRANSFER_HIGH,
                          'ASR_REDUCTION_MIN': ASR_REDUCTION_MIN,
                          'ACC_LOSS_MAX': ACC_LOSS_MAX},
    'combos': combo_df.to_dict(orient='records'),
    'strict_mcdm': strict_df.to_dict(orient='records'),
    'extended_mcdm': extended_df.to_dict(orient='records'),
    'per_dataset_comparison': result_df.to_dict(orient='records'),
}
out_path = f'{OUT_DIR}/item1_mcdm_comparison_results.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print('\nSaved ->', out_path)
