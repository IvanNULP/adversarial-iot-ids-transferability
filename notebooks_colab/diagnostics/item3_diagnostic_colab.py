# ── DIAGNOSTIC CELL — check what's available on Drive before building
# the Gaussian-noise-injection defense pipeline (Item 3) ──

from google.colab import drive
drive.mount('/content/drive')

import os

BASE = '/content/drive/MyDrive/adversarial_iot_paper'

for sub in ['models', 'results/section6_v2', 'results/section7_v2', 'data']:
    path = f'{BASE}/{sub}'
    print(f'--- {sub} ---')
    if os.path.isdir(path):
        entries = os.listdir(path)
        print(f'{len(entries)} entries:')
        for e in sorted(entries)[:30]:
            full = os.path.join(path, e)
            size = os.path.getsize(full) if os.path.isfile(full) else '(dir)'
            print(f'  {e}  {size}')
    else:
        print('MISSING')
    print()

# Specifically check whether raw adversarial example arrays (X_adv) or
# perturbation deltas were saved anywhere, vs. only scalar results (asr, acc)
print('--- Searching for any .npy files anywhere under adversarial_iot_paper/results ---')
for root, dirs, files in os.walk(f'{BASE}/results'):
    for f in files:
        if f.endswith('.npy'):
            full = os.path.join(root, f)
            print(f'  {full}  ({os.path.getsize(full)} bytes)')

# Check defense_results_v2.json structure (already used for Feature Squeezing)
# to confirm the exact key format we should mirror for Gaussian-noise results
import json
S7 = f'{BASE}/results/section7_v2'
if os.path.exists(f'{S7}/defense_results_v2.json'):
    with open(f'{S7}/defense_results_v2.json') as f:
        defense = json.load(f)
    print(f'\ndefense_results_v2.json: {len(defense)} entries')
    for i, (k, v) in enumerate(defense.items()):
        print(repr(k), '->', v)
        if i >= 4:
            break
