## What this repository contains

This repository reproduces all experiments reported in the paper. The code is organized into three Google Colab notebooks that should be run in order:

1. **Section6_Baseline_v2_CLEAN.ipynb** — preprocessing pipeline and baseline classifier training
2. **Section7_Adversarial_v2_CLEAN.ipynb** — adversarial attack generation and transferability evaluation
3. **Section7_Figures_v2_CLEAN.ipynb** — figure generation (Figures 6–12)

All intermediate results are saved to Google Drive after each step, so the notebooks can be interrupted and resumed without losing progress.

A seven-part extension package (novelty items 1–7, see below) builds on these three notebooks' outputs and lives in a separate `notebooks_colab/` folder.

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

**Step 5 (optional).** Run the novelty package (Items 1–7, see below) after Steps 1–4 complete — each script reuses the results and trained models produced above.

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

Two candidate defenses are evaluated on CIC-IDS 2017 and UNSW-NB15 at ε = 0.05:

- **Feature Squeezing** (4-bit depth + sliding window of 3) — the original defense candidate.
- **Gaussian noise injection** (additive zero-mean noise, σ = 0.05) — added as a second candidate in the novelty package (Item 3, see below), so that AARDF's defense-selection step evaluates more than one option.

Adversarial training is left as future work.

---

## Novelty package (Items 1–7)

In addition to the core AARDF/SATA benchmark above, this repository includes a seven-part extension package under [`notebooks_colab/`](notebooks_colab/), run after the three main notebooks:

| # | Item | What it adds |
|---|---|---|
| 1 | Weighted-MCDM comparison | Compares AARDF against a weighted multi-criteria baseline built from the same evidence metrics; the two disagree on 50–56% of deployment decisions |
| 2 | Pareto-optimality formalization | Formally characterizes AARDF's threshold selection as a bi-objective trade-off; proves a non-trivial Pareto frontier exists (10 of 25 configurations are non-dominated) |
| 3 | Second defense mechanism | Adds Gaussian noise injection as a second candidate defense alongside Feature Squeezing |
| 4 | Rule-induction comparison | Fits a small decision tree to the 16 dataset–architecture combinations and shows it is unstable under leave-one-out cross-validation at this sample size |
| 5 | Formal verification | Proves AARDF's decision procedure is a total, boundedly-terminating function |
| 6 | Decision-margin / manipulation-resistance | Applies AARDF's own evidentiary logic to AARDF itself: quantifies how close each decision lies to its governing threshold and how easily it could be perturbed |
| 7 | Counterfactual explanations | For each REASSESS outcome, computes the minimal evidence-metric change that would flip it to DEPLOY |

See [`notebooks_colab/README.md`](notebooks_colab/README.md) for a per-script breakdown, exact output paths, and reproducibility notes.

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
| AARDF vs. weighted-MCDM | Disagree on 50–56% of deployment decisions (Item 1) |
| Pareto-optimal thresholds | 10 of 25 configurations non-dominated; primary configuration is principled but not itself Pareto-optimal (Item 2) |
| Most fragile decision channel | Robustness Score: 2 of 16 decisions flip under ±1% perturbation (Item 6) |
| Counterfactual sparsity | 7 of 8 REASSESS outcomes reversible via a single-metric change (Item 7) |

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

See `requirements.txt` for exact versions. The novelty package (Items 1–7) uses only packages already listed here plus the Python standard library — no additional dependencies.

---

## Repository structure

```
├── Section6_Baseline_v2_CLEAN.ipynb
├── Section7_Adversarial_v2_CLEAN.ipynb
├── Section7_Figures_v2_CLEAN.ipynb
├── notebooks_colab/                          ← novelty package (Items 1-7)
│   ├── README.md
│   ├── item1_mcdm_colab.py
│   ├── item2_pareto_colab.py
│   ├── item2_pareto_figure_colab.py
│   ├── item3_gaussian_noise_colab.py
│   ├── item3_figure_colab.py
│   ├── item4_rule_induction_colab.py
│   ├── item6_decision_margin_colab.py
│   ├── item7_counterfactuals_colab.py
│   ├── figure15_update_mcdm_colab.py
│   └── diagnostics/
│       ├── diagnostic_cd_check.py
│       └── item3_diagnostic_colab.py
├── results/
│   ├── section6_v2/ ... section7_v2/          ← core benchmark results
│   └── section9_item1_mcdm/ ... section14_item7_counterfactuals/  ← Items 1-7 results
├── requirements.txt
└── README.md
```

---

## Citation

If you use this code or results in your work, please cite:

```
Opirskyy I., Susukailo [initial?]., Tyshyk [initial?]., Bortnik [initial?].,
Nakonechnyy [initial?]., Parkhuts [initial?]., Kostiak [initial?]. 
A Knowledge-Based Method and Formal Model (AARDF) for Adversarial 
Robustness Assessment and Defense Selection in Cross-Dataset IoT 
Intrusion Detection Systems [software]. GitHub; 2026. Available from: 
https://github.com/IvanNULP/adversarial-iot-ids-transferability
```

*(TODO before publishing: fill in each co-author's first-name initial(s)
above and confirm author order matches the paper's final author list.
I only have surnames from our prior work on the manuscript, not initials
— please fill these in yourself rather than trust a guess.)*

---

## License

Code is released under the MIT License. Dataset usage is subject to the respective dataset providers' terms.
