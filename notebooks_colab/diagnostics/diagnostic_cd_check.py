# ── DIAGNOSTIC CELL — run this before trusting the ASR_cross NaN pattern ──
# Checks whether XGBoost/CNN cross-dataset entries genuinely don't exist,
# or are silently missed due to a model-name / direction mismatch.

import json

S7 = '/content/drive/MyDrive/adversarial_iot_paper/results/section7_v2'
with open(f'{S7}/cd_results_v2.json') as f:
    cd = json.load(f)

print(f"Total entries in cd_results_v2.json: {len(cd)}")

# 1. What model name strings actually appear?
models_seen = sorted(set(v['model'] for v in cd.values()))
print("\nDistinct 'model' values in cd_results_v2.json:", models_seen)

# 2. What dataset name strings actually appear (src and tgt)?
src_seen = sorted(set(v['src_ds'] for v in cd.values()))
tgt_seen = sorted(set(v['tgt_ds'] for v in cd.values()))
print("Distinct 'src_ds' values:", src_seen)
print("Distinct 'tgt_ds' values:", tgt_seen)

# 3. Full breakdown: for each (src_ds, tgt_ds, model) combo among
#    CIC-IDS 2017 / UNSW-NB15, how many entries exist?
print("\n--- Coverage breakdown for CIC-IDS 2017 <-> UNSW-NB15 ---")
pairs = [('CIC-IDS 2017', 'UNSW-NB15'), ('UNSW-NB15', 'CIC-IDS 2017')]
for src, tgt in pairs:
    for m in models_seen:
        vals = [v['asr_tgt'] for v in cd.values()
                if v['src_ds'] == src and v['tgt_ds'] == tgt and v['model'] == m]
        print(f"  src={src:15s} tgt={tgt:15s} model={m:10s} -> {len(vals)} entries"
              + (f", max asr_tgt={max(vals):.4f}" if vals else ""))

# 4. Sample a few raw entries to eyeball exact key spelling/casing
print("\n--- Sample raw entries (first 10) ---")
for i, (k, v) in enumerate(cd.items()):
    print(repr(k), '->', v)
    if i >= 9:
        break
