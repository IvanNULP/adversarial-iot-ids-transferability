[README (1).md](https://github.com/user-attachments/files/27852138/README.1.md)

---
## What this repository contains

This repository reproduces all experiments reported in the paper. The code is organized into three Google Colab notebooks that should be run in order:

1. **Section6_Baseline_v2_C.ipynb** — preprocessing pipeline and baseline classifier training
2. **Section7_Adversarial_v2_C.ipynb** — adversarial attack generation and transferability evaluation
3. **Section7_Figures_v2_C.ipynb** — figure generation (Figures 6–12)

All intermediate results are saved to Google Drive after each step, so the notebooks can be interrupted and resumed without losing progress.

---

## Datasets

The experiments use four publicly available datasets:

| Dataset | Year | Source | Access |
|---|---|---|---|
| CIC-IDS 2017 | 2017 | University of New Brunswick | [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) |
| UNSW-NB15 | 2015 | UNSW Canberra | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| Gotham IoT 2025 | 2025 | Belarbi et al. | [Zenodo](https://doi.org/10.5281/zenodo.14502760) |
| CIC-YNU-IoTMal 2026 | 2026 | UNB CIC + Yunnan University | [UNB CIC](https://www.unb.ca/cic/datasets/) |

Download each dataset and place them under `data/` on your Google Drive:

```
adversarial_iot_paper/
└── data/
    ├── cicids2017/
    │   └── MachineLearningCVE/       ← CSV files from UNB CIC
    ├── unsw_nb15/
    │   ├── UNSW-NB15_1.csv           ← four CSV files
    │   ├── UNSW-NB15_2.csv
    │   ├── UNSW-NB15_3.csv
    │   ├── UNSW-NB15_4.csv
    │   └── NUSW-NB15_features.csv    ← feature names file
    ├── gotham2025/
    │   └── *.parquet                 ← parquet files from Zenodo
    └── cic_ynu_iotmal2026/
        ├── arm/pcap.parquet
        ├── mips/pcap.parquet
        ├── mipsel/pcap.parquet
        └── x86/pcap.parquet
```

---

## How to run

All notebooks are designed for Google Colab with GPU acceleration (tested on A100, 40 GB VRAM). A Colab Pro or Pro+ subscription is recommended for the adversarial experiments.

**Step 1.** Upload the notebooks to Google Colab or open them directly from this repository.

**Step 2.** Mount your Google Drive and place the datasets under `adversarial_iot_paper/data/` as shown above.

**Step 3.** Run the notebooks in order:

```
Section6_Baseline_v2_CLEAN.ipynb   →   Section7_Adversarial_v2_CLEAN.ipynb   →   Section7_Figures_v2_CLEAN.ipynb
```

Each notebook checks which steps have already been completed and skips them, so restarting after a crash is safe.

**Step 4.** Results are saved to Google Drive under:

```
adversarial_iot_paper/
├── results/
│   ├── section6_v2/     ← Table 6, processed arrays, scalers
│   └── section7_v2/     ← wb_results_v2.json, ca_results_v2.json, etc.
├── models_v2/           ← trained classifiers (.joblib, .pt)
└── figures/
    ├── section6_v2/     ← Figure 5 (ROC curves)
    └── section7_v2/     ← Figures 6–12
```

---

## Preprocessing pipeline

The pipeline follows a strict no-leakage order:

```
1. Data cleaning          → remove NaN, ±Inf, duplicates
2. Stratified split       → 70/30, seed=42  ← executed FIRST
3. Variance filter        → fit on X_train only
4. Min-Max normalisation  → scaler.fit(X_train), transform both
5. Correlation filter     → |r| < 0.01, computed on X_train only
6. SMOTE oversampling     → applied to X_train only
```

The test set is never used during any fitting step.

---

## Adversarial attacks

Three attack methods are evaluated using [IBM Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox):

| Attack | Type | ε levels | Notes |
|---|---|---|---|
| FGSM | Gradient-sign, one-step | 0.01, 0.05, 0.1 | L∞ norm |
| PGD | Iterative, 40 steps | 0.01, 0.05, 0.1 | L∞ norm, step=ε/4 |
| C&W | Optimization-based | 0.01, 0.05, 0.1 | L2 norm, 500 iters, 200-sample subset |

RF and XGBoost are attacked via a CNN surrogate model (black-box scenario), since they do not have analytical gradients.

---

## Cross-dataset transfer

Cross-dataset adversarial transfer is evaluated between CIC-IDS 2017 and UNSW-NB15 using a **delta transfer protocol**: perturbation vectors computed on source samples are applied to target samples. Eight semantically aligned flow-level features shared between the two datasets form the aligned subspace.

Gotham IoT 2025 and CIC-YNU-IoTMal 2026 participate in the multi-dataset and cross-architecture evaluation but not in the cross-dataset transfer due to incompatible feature spaces.

---

## Defense evaluation

Feature Squeezing (4-bit depth + sliding window of 3) is evaluated on CIC-IDS 2017 and UNSW-NB15 at ε = 0.05. Adversarial training is left as future work.

---

## Key results

| Finding | Result |
|---|---|
| Most robust model | RF (RS = 0.711) |
| Least robust model | CNN (RS = 0.438) |
| Attack hierarchy | C&W > PGD > FGSM (Wilcoxon p < 0.0001, n=24 matched triplets) |
| Max cross-arch TR | MLP→RF = 1.85 on UNSW-NB15 (ε = 0.01) |
| Max cross-dataset ASR | XGB UNSW→CIC = 0.555 (FGSM, ε = 0.1) |
| FS effect (CIC) | −30 to −56% ASR for MLP and CNN |
| FS effect (UNSW) | Ineffective or counterproductive |

---

## Environment

```
Python 3.10
scikit-learn 1.4
xgboost 2.0
torch 2.0
adversarial-robustness-toolbox 1.17
imbalanced-learn 0.12
shap 0.45
pandas 2.1
numpy 1.26
matplotlib 3.8
```

See `requirements.txt` for exact versions.

---

## Repository structure

```
├── Section6_Baseline_v2_CLEAN.ipynb
├── Section7_Adversarial_v2_CLEAN.ipynb
├── Section7_Figures_v2_CLEAN.ipynb
├── requirements.txt
└── README.md
```

---

## Citation

If you use this code or results in your work, please cite:

```
Opirskyy I. adversarial-iot-ids-transferability [Internet]. 
GitHub; 2026. Available from: 
https://github.com/IvanNULP/adversarial-iot-ids-transferability
```

---

## License

Code is released under the MIT License. Dataset usage is subject to the respective dataset providers' terms.
