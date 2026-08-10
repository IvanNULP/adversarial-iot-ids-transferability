# =============================================================
# ITEM 3 — Second defense mechanism: Gaussian noise injection /
# input randomization (NOT adversarial training). Mirrors the
# Feature Squeezing evaluation (CELL 9 of Section7_Adversarial_v2)
# exactly in structure, reusing the identical FGSM/PGD generation
# code, so results are directly comparable.
# Copy each CELL separately into Google Colab.
# =============================================================

# ── CELL 0: Mount Drive + resolve MODELS_DIR (models vs models_v2) ──
from google.colab import drive
drive.mount('/content/drive')
get_ipython().system('pip install adversarial-robustness-toolbox xgboost -q')

import os, json, warnings, gc
import numpy as np, pandas as pd
import joblib, torch, torch.nn as nn
from sklearn.metrics import accuracy_score
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
warnings.filterwarnings('ignore')

BASE     = '/content/drive/MyDrive/adversarial_iot_paper'
SEC6_DIR = os.path.join(BASE, 'results', 'section6_v2')
SEC7_DIR = os.path.join(BASE, 'results', 'section7_v2')
OUT_DIR  = os.path.join(BASE, 'results', 'section11_item3_gaussian_noise')
os.makedirs(OUT_DIR, exist_ok=True)

# Resolve the correct models folder (notebook says 'models_v2', but it may
# actually be saved as 'models' -- check both, use whichever exists).
candidates = [os.path.join(BASE, 'models_v2'), os.path.join(BASE, 'models')]
MODELS_DIR = next((c for c in candidates if os.path.isdir(c)), None)
print('MODELS_DIR resolved to:', MODELS_DIR)
if MODELS_DIR is None:
    raise RuntimeError('Neither models_v2 nor models folder found -- check Drive path.')
print('Sample contents:', sorted(os.listdir(MODELS_DIR))[:8])

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DS_TAGS = {
    'CIC-IDS 2017'       : 'CIC-IDS_2017',
    'UNSW-NB15'          : 'UNSW-NB15',
}  # matches the FS evaluation scope (CIC-IDS 2017 + UNSW-NB15 only)


# ── CELL 1: Model classes + loaders (identical to Section7_Adversarial_v2) ──
class CNN1D(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,128,3,padding=1), nn.BatchNorm1d(128),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128,64,3,padding=1), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,32,3,padding=1), nn.BatchNorm1d(32),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(32,128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes)
        )
    def forward(self, x): return self.head(self.conv(x.unsqueeze(1)))

class TorchMLP(nn.Module):
    def __init__(self, n_feat, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat,256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,128),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,64),    nn.ReLU(),
            nn.Linear(64, n_cls)
        )
    def forward(self, x): return self.net(x)

def make_art_clf(model, n_feat, n_cls):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    return PyTorchClassifier(
        model=model, loss=nn.CrossEntropyLoss(), optimizer=opt,
        input_shape=(n_feat,), nb_classes=n_cls,
        clip_values=(0.0, 1.0),
        device_type='gpu' if DEVICE=='cuda' else 'cpu')

def predict_batched(model, X, batch=4096):
    model.eval(); all_p = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.FloatTensor(X[i:i+batch]).to(DEVICE)
            all_p.append(torch.argmax(model(xb),1).cpu().numpy())
            del xb
    if DEVICE=='cuda': torch.cuda.empty_cache()
    return np.concatenate(all_p)

DATA = {}
for ds_name, tag in DS_TAGS.items():
    xtr = np.load(os.path.join(SEC6_DIR, f'X_tr_{tag}.npy'))
    xte = np.load(os.path.join(SEC6_DIR, f'X_te_{tag}.npy'))
    ytr = np.load(os.path.join(SEC6_DIR, f'y_tr_{tag}.npy'))
    yte = np.load(os.path.join(SEC6_DIR, f'y_te_{tag}.npy'))
    n_cls = len(np.unique(ytr))
    DATA[ds_name] = (xtr, xte, ytr, yte, n_cls)
    print(f'DATA/{ds_name}: te={len(xte):,} cls={n_cls} feat={xte.shape[1]}')

MODELS = {}
for ds_name, tag in DS_TAGS.items():
    MODELS[ds_name] = {}
    for mn in ['rf','xgb','mlp']:
        p = os.path.join(MODELS_DIR, f'{mn}_{tag}.joblib')
        if os.path.exists(p):
            MODELS[ds_name][mn.upper() if mn!='mlp' else 'MLP'] = joblib.load(p)
    pt, cfg = os.path.join(MODELS_DIR, f'cnn_{tag}.pt'), os.path.join(MODELS_DIR, f'cnn_{tag}_cfg.json')
    if os.path.exists(pt):
        with open(cfg) as f: c = json.load(f)
        m = CNN1D(c['n_features'], c['n_classes']).to(DEVICE)
        m.load_state_dict(torch.load(pt, map_location=DEVICE)); m.eval()
        MODELS[ds_name]['CNN'] = m
    print(f'MODELS/{ds_name}: {list(MODELS[ds_name].keys())}')

TORCH_MLPS = {}
for ds_name, tag in DS_TAGS.items():
    pt_p, cfg_p = os.path.join(MODELS_DIR, f'torch_mlp_{tag}.pt'), os.path.join(MODELS_DIR, f'torch_mlp_{tag}_cfg.json')
    if os.path.exists(pt_p):
        with open(cfg_p) as f: c = json.load(f)
        m = TorchMLP(c['n_features'], c['n_classes']).to(DEVICE)
        m.load_state_dict(torch.load(pt_p, map_location=DEVICE)); m.eval()
        TORCH_MLPS[ds_name] = m
        print(f'TorchMLP/{ds_name}: loaded')
    else:
        print(f'WARNING: torch_mlp not found for {ds_name} -- MLP evaluation will be skipped.')


# ── CELL 2: Gaussian noise defense + evaluation loop ──
# Mirrors CELL 9 (Feature Squeezing) exactly in structure; EPS_DEF=0.05
# matches the FS evaluation (Figure 14 caption: "eps = 0.05").
GN_SIGMA = 0.05  # std of additive Gaussian noise, same scale as eps -- confirm before final run

def gaussian_noise_defense(X, sigma=GN_SIGMA, rng=None):
    rng = rng or np.random.default_rng(SEED)
    noise = rng.normal(0, sigma, size=X.shape).astype(np.float32)
    return np.clip(X + noise, 0, 1).astype(np.float32)

wb = json.load(open(os.path.join(SEC7_DIR, 'wb_results_v2.json')))
gn_results = {}
EPS_DEF = 0.05
rng = np.random.default_rng(SEED)

for ds_name in ['CIC-IDS 2017', 'UNSW-NB15']:
    X_tr, X_te, y_tr, y_te, n_cls = DATA[ds_name]
    n_feat = X_te.shape[1]
    n_samp = min(3000, len(X_te))
    idx = np.random.choice(len(X_te), n_samp, replace=False)
    Xb, yb = X_te[idx], y_te[idx]

    for mn in ['CNN', 'MLP', 'RF', 'XGBoost']:
        for atk_name in ['FGSM', 'PGD']:
            key = f'{ds_name}|{mn}|{atk_name}|{EPS_DEF}'
            if key not in wb or 'error' in wb[key]:
                continue

            if mn in ['MLP', 'CNN']:
                model = TORCH_MLPS.get(ds_name) if mn == 'MLP' else MODELS[ds_name].get('CNN')
                if model is None: continue
                art_clf = make_art_clf(model, n_feat, n_cls)
            else:
                cnn_s = MODELS[ds_name].get('CNN')
                if cnn_s is None: continue
                art_clf = make_art_clf(cnn_s, n_feat, n_cls)
                model = MODELS[ds_name].get('RF' if mn == 'RF' else 'XGB')
                if model is None: continue

            try:
                if atk_name == 'FGSM':
                    atk = FastGradientMethod(art_clf, eps=EPS_DEF, eps_step=EPS_DEF, norm=np.inf, batch_size=512)
                else:
                    atk = ProjectedGradientDescent(art_clf, eps=EPS_DEF, eps_step=EPS_DEF/4, max_iter=40, norm=np.inf, batch_size=256)
                X_adv = atk.generate(x=Xb.astype(np.float32))
                X_adv_gn = gaussian_noise_defense(X_adv, rng=rng)
                X_clean_gn = gaussian_noise_defense(Xb, rng=rng)

                if mn in ['MLP', 'CNN']:
                    p_adv = predict_batched(model, X_adv)
                    p_adv_gn = predict_batched(model, X_adv_gn)
                    p_cln_gn = predict_batched(model, X_clean_gn)
                    p_clean = predict_batched(model, Xb)
                else:
                    p_adv = model.predict(X_adv)
                    p_adv_gn = model.predict(X_adv_gn)
                    p_cln_gn = model.predict(X_clean_gn)
                    p_clean = model.predict(Xb)

                asr_before = float(1 - accuracy_score(yb, p_adv))
                asr_gn     = float(1 - accuracy_score(yb, p_adv_gn))
                acc_clean  = float(accuracy_score(yb, p_clean))
                clean_gn   = float(accuracy_score(yb, p_cln_gn))

                gn_key = f'{ds_name}|{mn}|{atk_name}'
                gn_results[gn_key] = {
                    'ds': ds_name, 'model': mn, 'attack': atk_name,
                    'acc_clean': acc_clean, 'asr_before': asr_before,
                    'asr_gn': asr_gn, 'clean_gn': clean_gn, 'sigma': GN_SIGMA,
                }
                with open(os.path.join(OUT_DIR, 'gn_results_v2.json'), 'w') as f:
                    json.dump(gn_results, f, indent=2)
                print(f"{ds_name}|{mn}|{atk_name}: clean={acc_clean:.3f} "
                      f"ASR_before={asr_before:.3f} ASR_after_GN={asr_gn:.3f} clean_gn={clean_gn:.3f}")
            except Exception as e:
                print(f"ERROR {ds_name}|{mn}|{atk_name}: {e}")
            gc.collect()
            if DEVICE == 'cuda': torch.cuda.empty_cache()

print(f'\nSaved -> {OUT_DIR}/gn_results_v2.json  ({len(gn_results)} entries)')


# ── CELL 3: Sanity check -- compare recomputed asr_before against the
# already-published defense_results_v2.json (approximate match expected,
# NOT bit-exact -- see note on RNG-state divergence above) ──
with open(os.path.join(SEC7_DIR, 'defense_results_v2.json')) as f:
    fs_results = json.load(f)

print(f"{'key':35s} {'asr_before (FS run)':>20s} {'asr_before (GN run)':>20s} {'diff':>8s}")
for k, v in gn_results.items():
    fs_v = fs_results.get(k)
    if fs_v:
        diff = abs(v['asr_before'] - fs_v['asr_before'])
        print(f"{k:35s} {fs_v['asr_before']:>20.4f} {v['asr_before']:>20.4f} {diff:>8.4f}")
