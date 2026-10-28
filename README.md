# Impact of Hand X-Ray Preprocessing on Deep Feature Regression for Automated Bone Age Estimation

Code and data split for the conference paper:

> M. F. P. Mukhalis, S. M. Amulya, C. O. Tambunan, and J. Arifin, "Impact of Hand X-Ray Preprocessing on Deep Feature Regression for Automated Bone Age Estimation," 2026 (submitted).

The study evaluates how an image enhancement package and region of interest (ROI) cropping affect bone age regression when pretrained CNNs are used as **frozen** feature extractors.

## Pipeline

1. Downscale (longer side 512 px) and polarity correction (border median > 127).
2. Four preprocessing variants: `raw` (baseline), `enh` (min-max stretch + 3x3 median filter + CLAHE), `roi` (Otsu + morphology + largest component + 5% margin), `enh_roi`.
3. Letterbox to 224 x 224.
4. Frozen ImageNet CNNs from torchvision with global average pooling: ResNet50 (`IMAGENET1K_V2`), DenseNet121 (`IMAGENET1K_V1`), EfficientNet-B0 (`IMAGENET1K_V1`). CPU, FP32, evaluation mode.
5. Feature standardization + sex (0/1) -> ridge, SVR (RBF), MLP, PCA (128) + XGBoost.
6. Validation-weighted ensemble of the best regressor per backbone.

The preprocessing functions are in [`src/preprocessing.py`](src/preprocessing.py).

## Data

The radiographs come from the **RSNA Pediatric Bone Age** dataset and are **not redistributed** here. Download them from the official RSNA challenge page or Kaggle and keep the original file names (`<id>.png`).

`data/splits.csv` lists the image-level split used in the paper (70/15/15, stratified by age group and sex, seed 42):

| subset | images |
|---|---|
| train | 8,827 |
| val | 1,892 |
| test | 1,892 |

The 200 official RSNA test images were not used because their labels are not public. The dataset provides no patient identifiers, so the split is image-level.

## Notebooks (Google Colab)

| Notebook | Content |
|---|---|
| `01_preprocessing_and_feature_extraction.ipynb` | Preprocessing into four variants, image cache, CNN feature extraction |
| `02_regression_ensemble_and_analysis.ipynb` | Ablation, hyperparameter tuning, backbone x regressor comparison, ensemble, sex ablation, bootstrap CIs, figures |
| `03_revision_analyses.ipynb` | Ablation on ResNet50 and DenseNet121, repeated 5-fold CV, age-group CIs and balanced MAE, duplicate audit, computational cost, ROI quality audit |

Edit the `ROOT` / path variables in the first cell of each notebook. All selection steps use the validation subset; the test subset is used only for reporting.

## Environment

Python 3.13, PyTorch 2.11, torchvision 0.26, scikit-learn 1.6.1, XGBoost 3.4.1, NumPy 2.1.3, SciPy 1.16.3, OpenCV 5.0.0 (Google Colab, CPU runtime). See `requirements.txt`.

## Main results (test subset, MAE in months)

| Model | MAE |
|---|---|
| Mean predictor | 33.60 |
| DenseNet121 + ridge (selected by validation MAE) | 12.32 |
| Validation-weighted ensemble | 10.45 |

ROI cropping reduced the cross-validated MAE of DenseNet121 + ridge by 0.55 +/- 0.24 months (10 of 10 folds); the enhancement package showed no detectable benefit. An audit of 200 crops found fingertip truncation in 84.0% of them.

## License

Code: MIT License (see `LICENSE`). The RSNA images are subject to their own terms of use.
