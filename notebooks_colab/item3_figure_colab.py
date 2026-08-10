# =============================================================
# Figure — Gaussian Noise defense evaluation (Item 3), mirrors the
# style of the existing Figure 14 (Feature Squeezing).
# Saves as a NEW file — does not touch the existing Figure 14.
# Copy each CELL separately into Google Colab.
# =============================================================

# ── CELL 0: Mount Drive + Elsevier style setup ──
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
S11 = f'{BASE}/results/section11_item3_gaussian_noise'
FIG_DIR = f'{BASE}/figures/section6'
os.makedirs(FIG_DIR, exist_ok=True)

found = [f.name for f in fm.fontManager.ttflist if 'Liberation Sans' in f.name]
FONT = 'Liberation Sans' if found else 'DejaVu Sans'

matplotlib.rcParams.update({
    'font.family'      : FONT,
    'font.size'        : 8,
    'font.weight'      : 'normal',
    'axes.titleweight' : 'normal',
    'axes.labelweight' : 'normal',
    'axes.linewidth'   : 0.6,
    'axes.titlesize'   : 8,
    'axes.labelsize'   : 7.5,
    'xtick.labelsize'  : 6.5,
    'ytick.labelsize'  : 7,
    'legend.fontsize'  : 6.5,
    'legend.frameon'   : False,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'pdf.fonttype'     : 42,
    'ps.fonttype'      : 42,
})

FIG_W = 6.93  # 176 mm Elsevier double-column
DPI = 300


# ── CELL 1: Load Item 3 results ──
with open(f'{S11}/gn_results_v2.json') as f:
    gn = json.load(f)

print(f"{len(gn)} entries loaded.")
for k, v in gn.items():
    print(k, '->', v)


# ── CELL 2: Build the two-panel figure (CIC-IDS 2017 / UNSW-NB15) ──
model_order = ['RF', 'XGBoost', 'MLP', 'CNN']
attack_order = ['FGSM', 'PGD']

fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.6))

C_CLEAN = '#D5E8D4'
C_ASR_BEFORE = '#F8CECC'
C_ASR_AFTER = '#FFD97D'

for ax, ds_name, label in zip(axes, ['CIC-IDS 2017', 'UNSW-NB15'], ['(a)', '(b)']):
    labels, clean_vals, before_vals, after_vals = [], [], [], []
    for m in model_order:
        for a in attack_order:
            key = f'{ds_name}|{m}|{a}'
            if key not in gn:
                continue
            v = gn[key]
            labels.append(f'{m}\n{a}')
            clean_vals.append(v['acc_clean'])
            before_vals.append(v['asr_before'])
            after_vals.append(v['asr_gn'])

    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, clean_vals, w, label='Clean accuracy (before GN)', color=C_CLEAN, edgecolor='#444', linewidth=0.4)
    ax.bar(x,     before_vals, w, label='ASR before GN', color=C_ASR_BEFORE, edgecolor='#444', linewidth=0.4)
    ax.bar(x + w,  after_vals, w, label='ASR after GN', color=C_ASR_AFTER, edgecolor='#444', linewidth=0.4)

    for i, v in enumerate(after_vals):
        ax.text(x[i] + w, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=5.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{label} {ds_name}', fontsize=8, loc='left')
    if ax is axes[0]:
        ax.set_ylabel('Rate')

axes[0].legend(loc='upper left', ncol=1, fontsize=6, bbox_to_anchor=(0, 1.28))

plt.tight_layout()

out_png = f'{FIG_DIR}/figure_gn_defense_item3.png'
out_pdf = f'{FIG_DIR}/figure_gn_defense_item3.pdf'
plt.savefig(out_png, dpi=DPI, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()

print('Saved ->', out_png)
print('Saved ->', out_pdf)
