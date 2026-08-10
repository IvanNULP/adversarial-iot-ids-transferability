# =============================================================
# ITEM 6 — Decision margin and manipulation-resistance analysis.
# Reuses the exact aardf_decide() logic (with the XGBoost/XGB key
# fix from Item 1) and extends the candidate defense set to
# {Feature Squeezing, Gaussian noise injection} per Item 3/Section
# 4.9.1. No new adversarial-attack experiments -- pure post-hoc
# analysis of already-computed benchmark data.
# Copy each CELL separately into Google Colab.
# =============================================================

# ── CELL 0: Mount Drive + load all required data ──
from google.colab import drive
drive.mount('/content/drive')

import os, json
import pandas as pd
import numpy as np

BASE = '/content/drive/MyDrive/adversarial_iot_paper'
S6  = f'{BASE}/results/section6_v2'
S7  = f'{BASE}/results/section7_v2'
S11 = f'{BASE}/results/section11_item3_gaussian_noise'
OUT_DIR = f'{BASE}/results/section13_item6_decision_margin'
os.makedirs(OUT_DIR, exist_ok=True)

table6 = pd.read_csv(f'{S6}/table6_v2.csv')
with open(f'{S7}/wb_results_v2.json') as f: wb = json.load(f)
with open(f'{S7}/ca_results_v2.json') as f: ca = json.load(f)
with open(f'{S7}/cd_results_v2.json') as f: cd = json.load(f)
with open(f'{S7}/stat_results_v2.json') as f: stat = json.load(f)
with open(f'{S7}/defense_results_v2.json') as f: defense_fs = json.load(f)
with open(f'{S11}/gn_results_v2.json') as f: defense_gn = json.load(f)

datasets = ['CIC-IDS 2017', 'UNSW-NB15', 'Gotham IoT 2025', 'CIC-YNU-IoTMal 2026']
models = ['RF', 'XGBoost', 'MLP', 'CNN']

rs_block = stat['robustness_scores']
rs_values = {(d, m): [] for d in datasets for m in models}
for k, v in rs_block.items():
    d, m = v['ds'], v['model']
    if (d, m) in rs_values:
        rs_values[(d, m)].append(v['rs'])
rs_table = {key: (sum(vals)/len(vals) if vals else None) for key, vals in rs_values.items()}

CD_MODEL_ALIAS = {'XGBoost': 'XGB'}  # cd_results_v2.json / ca_results_v2.json quirk, fixed in Item 1

print('Data loaded. RS table sample:', list(rs_table.items())[:2])


# ── CELL 1: Reusable metric functions (identical to Item 1/2, XGBoost fix applied) ──
RS_LOW = 0.45
TR_HIGH = 1.5
ASR_TRANSFER_HIGH = 0.45
ASR_REDUCTION_MIN = 0.30
ACC_LOSS_MAX = 0.20

def clean_acc(dataset, model):
    row = table6[table6['Dataset'] == dataset]
    if row.empty: return None
    col = f'{model}_Acc'
    return float(row.iloc[0][col]) if col in table6.columns else None

def tr_max_into(dataset, model, ca_json):
    vals = [v['tr'] for v in ca_json.values() if v['ds'] == dataset and v['tgt'] == model]
    return max(vals) if vals else 0.0

def asr_cross_into(dataset, model, cd_json):
    if dataset not in ('CIC-IDS 2017', 'UNSW-NB15'):
        return None
    cd_model = CD_MODEL_ALIAS.get(model, model)
    other = 'UNSW-NB15' if dataset == 'CIC-IDS 2017' else 'CIC-IDS 2017'
    vals = [v['asr_tgt'] for v in cd_json.values()
            if v['src_ds'] == other and v['tgt_ds'] == dataset and v['model'] == cd_model]
    return max(vals) if vals else None

def fs_effect(dataset, model):
    reductions, acc_losses = [], []
    for v in defense_fs.values():
        if v['ds'] == dataset and v['model'] == model:
            if v['asr_before'] > 0:
                reductions.append((v['asr_before'] - v['asr_fs']) / v['asr_before'])
            if v['acc_clean']:
                acc_losses.append((v['acc_clean'] - v['clean_fs']) / v['acc_clean'])
    r = sum(reductions)/len(reductions) if reductions else None
    a = sum(acc_losses)/len(acc_losses) if acc_losses else None
    return r, a

def gn_effect(dataset, model):
    reductions, acc_losses = [], []
    for v in defense_gn.values():
        if v['ds'] == dataset and v['model'] == model:
            if v['asr_before'] > 0:
                reductions.append((v['asr_before'] - v['asr_gn']) / v['asr_before'])
            if v['acc_clean']:
                acc_losses.append((v['acc_clean'] - v['clean_gn']) / v['acc_clean'])
    r = sum(reductions)/len(reductions) if reductions else None
    a = sum(acc_losses)/len(acc_losses) if acc_losses else None
    return r, a

CANDIDATE_DEFENSES = {'FS': fs_effect, 'GN': gn_effect}


# ── CELL 2: aardf_decide with optional overrides (for perturbation) and
# decision-path tracking (for decision-margin computation) ──
def aardf_decide(dataset, model, rs_override=None, tr_override=None, asr_override=None):
    rs = rs_override if rs_override is not None else rs_table.get((dataset, model))
    if rs is None:
        return 'UNKNOWN', {'path': []}
    trace = {'RS': rs, 'path': ['RS']}
    if rs < RS_LOW:
        trace['margin'] = rs - RS_LOW
        trace['binding_check'] = 'RS'
        return 'REASSESS', trace

    tr_max = tr_override if tr_override is not None else tr_max_into(dataset, model, ca)
    trace['TR_max'] = tr_max
    trace['path'].append('TR')
    flags = []
    if tr_max > TR_HIGH:
        flags.append('DATASET_SPECIFIC_SURROGATE_RISK')

    asr_cross = asr_override if asr_override is not None else asr_cross_into(dataset, model, cd)
    trace['ASR_cross'] = asr_cross
    if asr_cross is not None:
        trace['path'].append('ASR_cross')
        if asr_cross > ASR_TRANSFER_HIGH:
            flags.append('CROSS_GENERATION_RISK')
    trace['flags'] = flags

    if flags:
        # binding check = the FLAG-FIRING criterion closest to its own
        # threshold (smallest |margin|), i.e. the most fragile one --
        # NOT simply the most negative raw value, which would instead
        # select whichever criterion failed by the LARGEST margin.
        margins = {}
        if 'DATASET_SPECIFIC_SURROGATE_RISK' in flags:
            margins['TR'] = TR_HIGH - tr_max
        if 'CROSS_GENERATION_RISK' in flags:
            margins['ASR_cross'] = ASR_TRANSFER_HIGH - asr_cross
        trace['binding_check'] = min(margins, key=lambda k: abs(margins[k]))
        trace['margin'] = margins[trace['binding_check']]

        for fname, fn in CANDIDATE_DEFENSES.items():
            reduction, acc_loss = fn(dataset, model)
            trace[f'{fname}_reduction'] = reduction
            trace[f'{fname}_acc_loss'] = acc_loss
            if reduction is not None and acc_loss is not None and reduction >= ASR_REDUCTION_MIN and acc_loss <= ACC_LOSS_MAX:
                return f'DEPLOY_WITH_DEFENSE({fname})', trace
        return 'REASSESS', trace

    # No flag fired: margin = distance to the NEAREST threshold this combo could
    # have failed (the tightest of the checks actually evaluated)
    margins = {'TR': TR_HIGH - tr_max}
    if asr_cross is not None:
        margins['ASR_cross'] = ASR_TRANSFER_HIGH - asr_cross
    trace['margin'] = min(margins.values())
    trace['binding_check'] = min(margins, key=margins.get)
    return 'DEPLOY', trace

def collapse_binary(decision):
    return 'REASSESS' if decision in ('REASSESS', 'UNKNOWN') else 'DEPLOY'


# ── CELL 3: Decision margin for all 16 combinations (unperturbed) ──
rows = []
for d in datasets:
    for m in models:
        decision, trace = aardf_decide(d, m)
        rows.append({
            'dataset': d, 'model': m, 'decision': decision,
            'binary': collapse_binary(decision),
            'RS': trace.get('RS'), 'TR_max': trace.get('TR_max'), 'ASR_cross': trace.get('ASR_cross'),
            'binding_check': trace.get('binding_check'), 'margin': trace.get('margin'),
        })
margin_df = pd.DataFrame(rows)
print(margin_df.to_string(index=False))


# ── CELL 4: Manipulation resistance -- perturb each defined metric by
# +/-1%, 2%, 5%, 10% (relative), one metric at a time, and check whether
# the collapsed binary decision flips ──
PERTURBATIONS = [0.01, 0.02, 0.05, 0.10]
records = []

for d in datasets:
    for m in models:
        base_decision, base_trace = aardf_decide(d, m)
        base_binary = collapse_binary(base_decision)
        rs0, tr0, asr0 = base_trace.get('RS'), base_trace.get('TR_max'), base_trace.get('ASR_cross')

        for metric, base_val in [('RS', rs0), ('TR_max', tr0), ('ASR_cross', asr0)]:
            if base_val is None:
                continue
            for p in PERTURBATIONS:
                for sign in [+1, -1]:
                    perturbed = base_val * (1 + sign * p)
                    kwargs = {}
                    if metric == 'RS': kwargs['rs_override'] = perturbed
                    elif metric == 'TR_max': kwargs['tr_override'] = perturbed
                    elif metric == 'ASR_cross': kwargs['asr_override'] = perturbed
                    new_decision, _ = aardf_decide(d, m, **kwargs)
                    new_binary = collapse_binary(new_decision)
                    records.append({
                        'dataset': d, 'model': m, 'metric': metric, 'perturbation': p,
                        'sign': sign, 'base_binary': base_binary, 'new_binary': new_binary,
                        'flipped': base_binary != new_binary,
                    })

perturb_df = pd.DataFrame(records)
print(f"\n{len(perturb_df)} perturbation trials run.")

summary = perturb_df.groupby('perturbation')['flipped'].agg(['sum', 'count', 'mean'])
print("\n=== Flip rate by perturbation magnitude (all metrics/signs pooled) ===")
print(summary)

print("\n=== Flip rate by metric, at each perturbation magnitude ===")
print(perturb_df.groupby(['metric', 'perturbation'])['flipped'].agg(['sum','count','mean']))

print("\n=== Which (dataset, model) combinations flip at 5% or less? ===")
flips_5 = perturb_df[(perturb_df['perturbation'] <= 0.05) & (perturb_df['flipped'])]
print(flips_5[['dataset','model','metric','perturbation','sign']].drop_duplicates().to_string(index=False))


# ── CELL 5: Save everything ──
out = {
    'thresholds': {'RS_LOW': RS_LOW, 'TR_HIGH': TR_HIGH, 'ASR_TRANSFER_HIGH': ASR_TRANSFER_HIGH},
    'decision_margins': margin_df.to_dict(orient='records'),
    'perturbation_trials': perturb_df.to_dict(orient='records'),
    'flip_rate_by_perturbation': summary.reset_index().to_dict(orient='records'),
}
out_path = f'{OUT_DIR}/item6_decision_margin_results.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print('\nSaved ->', out_path)
