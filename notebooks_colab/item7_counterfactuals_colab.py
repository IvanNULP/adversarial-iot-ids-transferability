# =============================================================
# ITEM 7 — Counterfactual explanations (Wachter et al. [53]).
# For each REASSESS-flagged combination, find the minimal single-
# metric change that flips the decision to DEPLOY / DEPLOY_WITH_DEFENSE,
# verified by actually calling aardf_decide() rather than trusting
# raw margin arithmetic alone (a single-criterion fix is not always
# sufficient when multiple risk flags are simultaneously active --
# confirmed for CIC-IDS 2017/XGBoost in Item 6). Falls back to a
# joint two-metric counterfactual when no single-metric one exists.
# Reuses the exact aardf_decide() from Item 6 -- no new experiments.
# Copy each CELL separately into Google Colab.
# =============================================================

# ── CELL 0: Mount Drive + load all required data (identical to Item 6) ──
from google.colab import drive
drive.mount('/content/drive')

import os, json
import pandas as pd
import numpy as np
from itertools import combinations

BASE = '/content/drive/MyDrive/adversarial_iot_paper'
S6  = f'{BASE}/results/section6_v2'
S7  = f'{BASE}/results/section7_v2'
S11 = f'{BASE}/results/section11_item3_gaussian_noise'
OUT_DIR = f'{BASE}/results/section14_item7_counterfactuals'
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

CD_MODEL_ALIAS = {'XGBoost': 'XGB'}

RS_LOW = 0.45
TR_HIGH = 1.5
ASR_TRANSFER_HIGH = 0.45
ASR_REDUCTION_MIN = 0.30
ACC_LOSS_MAX = 0.20

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

def aardf_decide(dataset, model, rs_override=None, tr_override=None, asr_override=None):
    rs = rs_override if rs_override is not None else rs_table.get((dataset, model))
    if rs is None:
        return 'UNKNOWN', {}
    if rs < RS_LOW:
        return 'REASSESS', {}
    tr_max = tr_override if tr_override is not None else tr_max_into(dataset, model, ca)
    flags = []
    if tr_max > TR_HIGH:
        flags.append('DATASET_SPECIFIC_SURROGATE_RISK')
    asr_cross = asr_override if asr_override is not None else asr_cross_into(dataset, model, cd)
    if asr_cross is not None and asr_cross > ASR_TRANSFER_HIGH:
        flags.append('CROSS_GENERATION_RISK')
    if flags:
        for fname, fn in CANDIDATE_DEFENSES.items():
            reduction, acc_loss = fn(dataset, model)
            if reduction is not None and acc_loss is not None and reduction >= ASR_REDUCTION_MIN and acc_loss <= ACC_LOSS_MAX:
                return f'DEPLOY_WITH_DEFENSE({fname})', {}
        return 'REASSESS', {}
    return 'DEPLOY', {}

def collapse_binary(decision):
    return 'REASSESS' if decision in ('REASSESS', 'UNKNOWN') else 'DEPLOY'

EPS = 1e-6  # tiny margin past the threshold, avoids landing exactly on the boundary


# ── CELL 1: Identify all REASSESS combinations and their real metric values ──
reassess_combos = []
for d in datasets:
    for m in models:
        decision, _ = aardf_decide(d, m)
        if collapse_binary(decision) == 'REASSESS':
            rs = rs_table.get((d, m))
            tr = tr_max_into(d, m, ca) if rs is not None and rs >= RS_LOW else None
            asr = asr_cross_into(d, m, cd) if tr is not None else None
            reassess_combos.append({'dataset': d, 'model': m, 'RS': rs, 'TR_max': tr, 'ASR_cross': asr})

print(f"{len(reassess_combos)} REASSESS combinations found:")
for c in reassess_combos:
    print(c)


# ── CELL 2: Single-metric counterfactual search (verified, not just margin) ──
def try_single_metric_counterfactual(dataset, model, rs, tr, asr):
    results = {}
    # RS counterfactual: only meaningful if RS itself is the failing criterion
    if rs is not None and rs < RS_LOW:
        candidate_rs = RS_LOW + EPS
        decision, _ = aardf_decide(dataset, model, rs_override=candidate_rs)
        results['RS'] = {
            'candidate_value': candidate_rs,
            'relative_change': (candidate_rs - rs) / rs,
            'flips_to_deploy': collapse_binary(decision) == 'DEPLOY',
            'resulting_decision': decision,
        }
    # TR_max counterfactual: only meaningful if RS already passes
    if rs is not None and rs >= RS_LOW and tr is not None and tr > TR_HIGH:
        candidate_tr = TR_HIGH - EPS
        decision, _ = aardf_decide(dataset, model, tr_override=candidate_tr)
        results['TR_max'] = {
            'candidate_value': candidate_tr,
            'relative_change': (candidate_tr - tr) / tr,
            'flips_to_deploy': collapse_binary(decision) == 'DEPLOY',
            'resulting_decision': decision,
        }
    # ASR_cross counterfactual: only meaningful if defined and failing
    if rs is not None and rs >= RS_LOW and asr is not None and asr > ASR_TRANSFER_HIGH:
        candidate_asr = ASR_TRANSFER_HIGH - EPS
        decision, _ = aardf_decide(dataset, model, asr_override=candidate_asr)
        results['ASR_cross'] = {
            'candidate_value': candidate_asr,
            'relative_change': (candidate_asr - asr) / asr,
            'flips_to_deploy': collapse_binary(decision) == 'DEPLOY',
            'resulting_decision': decision,
        }
    return results

def try_joint_counterfactual(dataset, model, rs, tr, asr):
    # Only called when no single-metric counterfactual sufficed.
    # Try all pairs (and the triple) of the criteria that are actually failing.
    failing = {}
    if rs is not None and rs < RS_LOW:
        failing['RS'] = RS_LOW + EPS
    if tr is not None and tr > TR_HIGH:
        failing['TR_max'] = TR_HIGH - EPS
    if asr is not None and asr > ASR_TRANSFER_HIGH:
        failing['ASR_cross'] = ASR_TRANSFER_HIGH - EPS
    names = list(failing.keys())
    for r in range(2, len(names) + 1):
        for combo in combinations(names, r):
            kwargs = {}
            if 'RS' in combo: kwargs['rs_override'] = failing['RS']
            if 'TR_max' in combo: kwargs['tr_override'] = failing['TR_max']
            if 'ASR_cross' in combo: kwargs['asr_override'] = failing['ASR_cross']
            decision, _ = aardf_decide(dataset, model, **kwargs)
            if collapse_binary(decision) == 'DEPLOY':
                return {'metrics': combo, 'resulting_decision': decision}
    return None

counterfactuals = []
for c in reassess_combos:
    d, m, rs, tr, asr = c['dataset'], c['model'], c['RS'], c['TR_max'], c['ASR_cross']
    single = try_single_metric_counterfactual(d, m, rs, tr, asr)
    viable_single = {k: v for k, v in single.items() if v['flips_to_deploy']}

    entry = {'dataset': d, 'model': m, 'single_metric_results': single}
    if viable_single:
        best_metric = min(viable_single, key=lambda k: abs(viable_single[k]['relative_change']))
        entry['counterfactual_type'] = 'single-metric'
        entry['minimal_metric'] = best_metric
        entry['minimal_relative_change'] = viable_single[best_metric]['relative_change']
        entry['resulting_decision'] = viable_single[best_metric]['resulting_decision']
    else:
        joint = try_joint_counterfactual(d, m, rs, tr, asr)
        entry['counterfactual_type'] = 'joint' if joint else 'none_found'
        if joint:
            entry['joint_metrics'] = joint['metrics']
            entry['resulting_decision'] = joint['resulting_decision']
    counterfactuals.append(entry)

for e in counterfactuals:
    print(f"\n{e['dataset']} / {e['model']}:")
    print(f"  type = {e['counterfactual_type']}")
    if e['counterfactual_type'] == 'single-metric':
        print(f"  minimal change: {e['minimal_metric']} by {e['minimal_relative_change']*100:.2f}% -> {e['resulting_decision']}")
    elif e['counterfactual_type'] == 'joint':
        print(f"  requires joint change in: {e['joint_metrics']} -> {e['resulting_decision']}")
    else:
        print("  NO counterfactual found even with all failing metrics corrected "
              "(defense evaluation itself is the limiting factor)")


# ── CELL 3: Save results ──
out = {
    'reassess_combinations': reassess_combos,
    'counterfactuals': counterfactuals,
}
out_path = f'{OUT_DIR}/item7_counterfactual_results.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print('\nSaved ->', out_path)
