# =============================================================
# ITEM 4 — Rule-induction comparison: small decision tree learned
# from the 16 dataset-architecture combinations vs. AARDF's
# expert-encoded threshold cascade. Explicitly frames n=16 as too
# small for a rigorous ML claim -- an argument FOR expert-encoded
# rules at this sample size, not a competing method.
# Reuses the 16-combo data already saved from Item 1 -- no new
# experiments. Copy each CELL separately into Google Colab.
# =============================================================

# ── CELL 0: Mount Drive + load the 16-combo data from Item 1 ──
from google.colab import drive
drive.mount('/content/drive')

import os, json
import numpy as np
import pandas as pd

BASE = '/content/drive/MyDrive/adversarial_iot_paper'
S9 = f'{BASE}/results/section9_item1_mcdm'
OUT_DIR = f'{BASE}/results/section12_item4_rule_induction'
os.makedirs(OUT_DIR, exist_ok=True)

with open(f'{S9}/item1_mcdm_comparison_results.json') as f:
    item1 = json.load(f)

df = pd.DataFrame(item1['combos'])
print(df.to_string(index=False))


# ── CELL 1: Prepare features (RS, TR_max, ASR_whitebox) and label ──
# ASR_whitebox is used (not ASR_cross) because it is defined for all
# 16 combinations; ASR_cross is only available for 6 (Item 1, Section
# 6.11) and would force dropping 10 of 16 samples here.
FEATURES = ['RS', 'TR_max', 'ASR_whitebox']

def collapse_binary(decision):
    return 'REASSESS' if decision in ('REASSESS', 'UNKNOWN') else 'DEPLOY'

df['label'] = df['aardf_decision'].apply(collapse_binary)

X = df[FEATURES].values
y = (df['label'] == 'DEPLOY').astype(int).values  # 1 = DEPLOY, 0 = REASSESS

print(f"\nn = {len(df)} samples, {y.sum()} DEPLOY / {(1-y).sum()} REASSESS")
print(df[['dataset','model'] + FEATURES + ['label']].to_string(index=False))


# ── CELL 2: Fit a small decision tree (resubstitution fit only --
# NOT a generalization estimate at n=16) ──
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

for depth in [2, 3]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X, y)
    pred = tree.predict(X)
    acc = accuracy_score(y, pred)
    print(f"\n=== Decision tree, max_depth={depth} ===")
    print(f"Resubstitution accuracy (fit on all 16, evaluated on the same 16): {acc:.3f}")
    print("NOTE: this is a fit-quality measure, not a generalization estimate.")
    print(export_text(tree, feature_names=FEATURES))


# ── CELL 3: Leave-one-out cross-validation (n=16 -> 16 folds) --
# this is the key diagnostic for small-sample instability ──
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
fold_results = []
for depth in [2, 3]:
    correct = 0
    trees_seen = []
    for train_idx, test_idx in loo.split(X):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X[train_idx], y[train_idx])
        pred = tree.predict(X[test_idx])[0]
        correct += int(pred == y[test_idx][0])
        trees_seen.append(export_text(tree, feature_names=FEATURES))
    loo_acc = correct / len(X)
    n_unique_trees = len(set(trees_seen))
    print(f"\nmax_depth={depth}: LOO-CV accuracy = {loo_acc:.3f} "
          f"({correct}/{len(X)}); {n_unique_trees} distinct tree structures "
          f"across the 16 leave-one-out folds (out of 16 possible)")
    fold_results.append({'depth': depth, 'loo_accuracy': loo_acc,
                          'n_distinct_trees': n_unique_trees})


# ── CELL 4: Compare induced split thresholds to AARDF's expert thresholds ──
tree_full = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_full.fit(X, y)
t = tree_full.tree_
print("\n=== Induced split thresholds (max_depth=2, fit on all 16) ===")
for i in range(t.node_count):
    if t.children_left[i] != t.children_right[i]:  # internal node
        feat = FEATURES[t.feature[i]]
        thresh = t.threshold[i]
        print(f"Node {i}: split on {feat} at {thresh:.4f}")

print("\nAARDF's expert-encoded thresholds (Algorithm 1, Section 4.9):")
print("  RS_low = 0.45")
print("  TR_high = 1.5")
print("  ASR_transfer_high = 0.45")


# ── CELL 5: Save results ──
out = {
    'features': FEATURES,
    'n_samples': len(df),
    'n_deploy': int(y.sum()),
    'n_reassess': int((1-y).sum()),
    'loo_results': fold_results,
    'combos': df[['dataset','model'] + FEATURES + ['label']].to_dict(orient='records'),
}
out_path = f'{OUT_DIR}/item4_rule_induction_results.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print('\nSaved ->', out_path)
