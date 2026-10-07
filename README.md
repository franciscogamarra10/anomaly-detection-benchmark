# anomaly-detection-benchmark
Comprehensive cross-paradigm anomaly detection benchmark: from unsupervised PCA and PyTorch VAEs to TabNet, GBDT ensembles, and Stacked AutoML.
# Comprehensive Anomaly Detection Benchmark: Cardiotocography Dataset

An end-to-end anomaly detection and classification benchmark evaluated on the Cardiotocography (`cardio.mat`) dataset from the ODDS repository (1,831 samples, 21 continuous features, ≈ 9.6 % anomaly prevalence).

---

## 🔬 Benchmark Pipeline

1. **Preprocessing:** Stratified 70/30 train/test split; $z$-score standardization fitted exclusively on normal instances (`y == 0`) for unsupervised baselines.
2. **Linear Subspace:** PCA decoupled into Squared Prediction Error (SPE) and Hotelling’s $T^2$ with Jackson-Mudholkar limits.
3. **Non-Parametric Trees:** Isolation Forest tuned via Optuna (3-fold CV on PR-AUC).
4. **Deep Reconstruction:** H2O Deep AutoEncoder with trimmed re-training (top 20 % reconstruction error filtered).
5. **Generative Modeling:** PyTorch Tabular VAE trained with ELBO loss and evaluated via a hybrid score ($\text{MSE} + \beta \cdot \text{KL}$).
6. **Tabular Deep Learning:** PyTorch-TabNet (sequential attention, early stopping, validation-tuned $F_2$ threshold).
7. **Supervised GBDT:** LightGBM and CatBoost with isotonic calibration and $F_2$ threshold optimization (`TunedThresholdClassifierCV`).
8. **Automated Stacking:** AutoGluon (`best_quality` preset, multi-fold bagging + multi-layer stacking).
9. **Meta-Analysis:** Master comparison across ROC-AUC / PR-AUC and Spearman rank correlation matrix across model anomaly outputs.

---

## 📊 Key Results (Test Set)

> **Baseline (Random Guess PR-AUC):** ≈ 0.096 (9.6 % anomaly prevalence)

| Rank | Paradigm              | Model / Metric                              |   ROC-AUC  |   PR-AUC   |
| :--: | :-------------------- | :------------------------------------------ | :--------: | :--------: |
|  1   | Stacked AutoML        | AutoGluon (`WeightedEnsemble_L3`)           | **0.9970** | **0.9802** |
|  2   | Supervised GBDT       | Blended GBDT (LightGBM + CatBoost)          |   0.9957   |   0.9729   |
|  3   | Supervised GBDT       | LightGBM (Isotonic Calibrated)              |   0.9963   |   0.9721   |
|  4   | Supervised GBDT       | CatBoost (Isotonic Calibrated)              |   0.9849   |   0.9559   |
|  5   | Tabular Deep Learning | TabNet                                      |   0.9941   |   0.9498   |
|  6   | Non-Parametric Tree   | Isolation Forest (Optuna)                   |   0.9519   |   0.6727   |
|  7   | Linear Unsupervised   | PCA Combined Index ($\text{SPE}_z + T^2_z$) |   0.9413   |   0.6702   |
|  8   | Deep Reconstruction   | H2O AutoEncoder (Trimmed)                   |   0.9370   |   0.6634   |
|  9   | Generative Model      | PyTorch VAE (Hybrid Score)                  |   0.9098   |   0.5975   |

---

## 💡 Main Findings

* **Pathological Geometry:** In the unsupervised setting, Hotelling’s $T^2$ (ROC-AUC 0.9342 / PR-AUC 0.6437) significantly outperforms SPE (ROC-AUC 0.8776 / PR-AUC 0.6199). Pathological fetal states manifest primarily as **extreme leverage deviations within the dominant physiological subspace**, rather than orthogonal correlation collapses.
* **Supervised Ceiling:** When ground-truth labels are available, supervised models outperform unsupervised methods by > 30 percentage points in PR-AUC. AutoGluon delivers the highest ranking performance through automated multi-layer stacking.
* **Signal Diversity:** Spearman rank correlation confirms that TabNet remains moderately decorrelated ($r_s \approx 0.53$–$0.58$) from tree ensembles, making it an effective standalone alternative when native, instance-level attention masks are required for clinical explainability.
* **Strict Anti-Leakage Protocol:** All decision thresholds ($F_2$, $\beta = 2.0$) were determined strictly via out-of-fold predictions or validation sets before evaluation on the test set.

---

## 🛠️ Setup & Requirements

### System Requirements
* Python 3.10+
* Java Runtime Environment (JRE/JDK 8+) for H2O

### Python Environment
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn tabulate
pip install torch optuna lightgbm catboost pytorch-tabnet h2o autogluon.tabular
