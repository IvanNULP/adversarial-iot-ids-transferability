# =============================================================
# Figure 15 UPDATE — adds weighted-MCDM bar to the existing
# naive vs AARDF aggregate-ASR comparison (Section 6.11, Item 1).
# Saves under a NEW filename — does NOT overwrite the original,
# already-approved figure15.
# Copy each CELL separately into Google Colab.
# =============================================================

# ── CELL 0: Mount Drive + Elsevier style setup (matches existing figures) ──
from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.run(['apt-get', 'install', '-y', 'fonts-liberation'], capture_output=True)

import os, json
import matplotlib
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = '/content/drive/MyDrive/adversarial_iot_paper'
S9 = f'{BASE}/results/section9_item1_mcdm'
FIG_DIR = f'{BASE}/figures/section6'   # same folder as the original Figure 15
os.makedirs(FIG_DIR, exist_ok=True)

found = [f.name for f in fm.fontManager.ttflist if 'Liberation Sans' in f.name]
FONT = 'Liberation Sans' if found else 'DejaVu Sans'
print('Font:', FONT)

matplotlib.rcParams.update({
    'font.family'     : FONT,
    'font.size'       : 8,
    'font.weight'     : 'normal',
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'axes.linewidth'  : 0.6,
    'axes.titlesize'  : 8,
    'axes.labelsize'  : 7.5,
    'xtick.labelsize' : 7,
    'ytick.labelsize' : 7,
    'legend.fontsize' : 6.5,
    'legend.frameon'  : False,
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'pdf.fonttype'    : 42,
    'ps.fonttype'     : 42,
})

FIG_W = 6.93   # 176 mm Elsevier double-column
DPI = 300


# ── CELL 1: Load Item 1 results saved earlier ──
with open(f'{S9}/item1_mcdm_comparison_results.json') as f:
    item1 = json.load(f)

comp = item1['per_dataset_comparison']
for row in comp:
    print(row)


# ── CELL 2: Build the updated figure — 3 bars per dataset ──
datasets = [r['dataset'] for r in comp]
naive_asr = [r['naive_ASR'] for r in comp]
aardf_asr = [r['aardf_ASR'] for r in comp]
mcdm_asr  = [r['mcdm_ASR_extended'] for r in comp]   # extended: covers all 4 datasets
naive_model = [r['naive_model'] for r in comp]
aardf_model = [r['aardf_model'] for r in comp]
mcdm_model  = [r['mcdm_model_extended'] for r in comp]

x = np.arange(len(datasets))
width = 0.26

fig, ax = plt.subplots(figsize=(FIG_W, 3.4))

C_NAIVE = '#B0B0B0'
C_AARDF = '#2878B5'
C_MCDM  = '#C82423'

b1 = ax.bar(x - width, naive_asr, width, label='Naive (accuracy-only)', color=C_NAIVE, edgecolor='#444', linewidth=0.5)
b2 = ax.bar(x,         aardf_asr, width, label='AARDF-guided',          color=C_AARDF, edgecolor='#444', linewidth=0.5)
b3 = ax.bar(x + width,  mcdm_asr, width, label='Weighted-MCDM',         color=C_MCDM,  edgecolor='#444', linewidth=0.5)

# Model-name labels above each bar (matches the existing Figure 15 style)
for bars, models in [(b1, naive_model), (b2, aardf_model), (b3, mcdm_model)]:
    for rect, m in zip(bars, models):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.012, m,
                 ha='center', va='bottom', fontsize=6, rotation=0)

ax.set_ylabel('Mean white-box ASR')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=7)
ax.set_ylim(0, max(max(naive_asr), max(aardf_asr), max(mcdm_asr)) * 1.22)
ax.legend(loc='upper left', ncol=3, bbox_to_anchor=(0, 1.18))

plt.tight_layout()

# Save under a NEW filename — original figure15.* is left untouched
out_png = f'{FIG_DIR}/figure15_update_mcdm.png'
out_pdf = f'{FIG_DIR}/figure15_update_mcdm.pdf'
plt.savefig(out_png, dpi=DPI, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()

print('Saved ->', out_png)
print('Saved ->', out_pdf)
