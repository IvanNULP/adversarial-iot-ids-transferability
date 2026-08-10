# =============================================================
# Figure — Pareto-frontier of AARDF's threshold selection (Item 2)
# Scatter of all 25 (RS_low, TR_high) configurations in objective
# space: ASR_gap (x) vs risk_exposure (y), both minimized.
# Pareto-optimal points highlighted; primary configuration marked.
# Saves as a NEW file — does not touch any existing approved figure.
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
import pandas as pd

BASE = '/content/drive/MyDrive/adversarial_iot_paper'
S10 = f'{BASE}/results/section10_item2_pareto'
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
    'xtick.labelsize'  : 7,
    'ytick.labelsize'  : 7,
    'legend.fontsize'  : 6.5,
    'legend.frameon'   : False,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'pdf.fonttype'     : 42,
    'ps.fonttype'      : 42,
})

FIG_W = 3.55  # 90 mm Elsevier single-column
DPI = 300


# ── CELL 1: Load Item 2 results saved earlier ──
with open(f'{S10}/item2_pareto_results.json') as f:
    item2 = json.load(f)

df = pd.DataFrame(item2['grid'])
print(df.to_string(index=False))


# ── CELL 2: Build the Pareto-frontier scatter plot ──
fig, ax = plt.subplots(figsize=(FIG_W, 3.0))

dominated = df[~df['pareto_optimal']]
pareto = df[df['pareto_optimal']].sort_values('ASR_gap')

ax.scatter(dominated['ASR_gap'], dominated['risk_exposure'],
           s=22, facecolor='#B0B0B0', edgecolor='#444', linewidth=0.5,
           label='Dominated configurations', zorder=2)
ax.scatter(pareto['ASR_gap'], pareto['risk_exposure'],
           s=30, facecolor='#C82423', edgecolor='#444', linewidth=0.5,
           label='Pareto-optimal', zorder=3)
ax.plot(pareto['ASR_gap'], pareto['risk_exposure'],
        color='#C82423', linewidth=0.7, linestyle='--', zorder=2)

# Mark the primary configuration used in Table 7/8 (RS_low=0.45, TR_high=1.5)
primary = df[(df['RS_LOW'] == 0.45) & (df['TR_HIGH'] == 1.5)]
ax.scatter(primary['ASR_gap'], primary['risk_exposure'],
           s=60, facecolor='none', edgecolor='#2878B5', linewidth=1.2,
           marker='D', label='Primary configuration\n(RS_low=0.45, TR_high=1.5)', zorder=4)

ax.axvline(0, color='#999', linewidth=0.5, linestyle=':', zorder=1)
ax.set_xlabel('ASR_gap  (AARDF mean ASR \u2212 naive mean ASR)')
ax.set_ylabel('Risk exposure  (1 \u2212 flagged fraction)')
ax.legend(loc='upper right', fontsize=6)

plt.tight_layout()

out_png = f'{FIG_DIR}/figure_pareto_frontier_item2.png'
out_pdf = f'{FIG_DIR}/figure_pareto_frontier_item2.pdf'
plt.savefig(out_png, dpi=DPI, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()

print('Saved ->', out_png)
print('Saved ->', out_pdf)
