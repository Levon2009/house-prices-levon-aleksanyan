# House Prices — Advanced Regression Techniques

## Author

**Levon Aleksanyan**

## Overview

Predicting residential sale prices.

The target is `SalePrice`. Submissions are scored by RMSE between the **log** of the
predicted and actual price, so all models here are trained on `log(SalePrice)`.

**Current result:** Ridge regression, 5-fold CV RMSE **0.1198**, Kaggle public score **0.14217**.

## Project structure

```
├── data/
│   ├── raw/                  # Original Kaggle files (train, test, data_description, sample_submission)
│   └── processed/            # Cleaned + encoded features written by 01_EDA.ipynb
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploration, cleaning, feature engineering, encoding
│   └── 02_Models.ipynb       # Model comparison, Ridge tuning, error analysis, submission
├── results/
│   ├── figures/              # Saved EDA plots
│   └── model_comparison.csv  # Per-fold CV scores for every model
└── submissions/
    ├── submission_ridge.csv  # Kaggle submission file
    └── kaggle_score.png      # Screenshot of the leaderboard score
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Run the notebooks **in order**, from top to bottom with a fresh kernel:

1. `notebooks/01_EDA.ipynb` writes `data/processed/train_processed.csv` and `test_processed.csv`.
2. `notebooks/02_Models.ipynb` reads them, writes `results/model_comparison.csv` and
   `submissions/submission_ridge.csv`.

## Approach

### 1. Exploration
- Plots of category counts, box plots against price, scatter plots for the area features, and a
  correlation heatmap of the top 15 numeric features.
- Two outliers were removed: `GrLivArea > 4000` with `SalePrice < 300000`.

### 2. Missing values

| Strategy | Columns |
|---|---|
| `"None"` (feature is absent) | Alley, MasVnrType, Bsmt*, FireplaceQu, Garage*, PoolQC, Fence, MiscFeature |
| `0` | MasVnrArea, GarageYrBlt, GarageArea, GarageCars, basement areas and bathrooms |
| Mode | MSZoning, Electrical, KitchenQual, Exterior1st/2nd, SaleType, Functional |
| Median | LotFrontage |

### 3. Feature engineering

| Feature | Definition |
|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` |
| `HouseAge` | `YrSold - YearBuilt` |
| `RemodAge` | `YrSold - YearRemodAdd` |
| `TotalBath` | `FullBath + 0.5·HalfBath + BsmtFullBath + 0.5·BsmtHalfBath` |

With Random Forest, these features improved 5-fold CV RMSE from 0.1383 to 0.1356.

The columns `Utilities` and `Street` were dropped because almost every house has the same value.

### 4. Encoding
- **Ordinal** columns (quality and condition ratings, basement exposure and finish, garage finish,
  fence, lot shape, slope, paved drive, central air) are mapped to ordered integers.
- **Nominal** columns are one-hot encoded. Test columns are aligned to train.

### 5. Modelling
Every model is evaluated with the same `KFold(n_splits=5, shuffle=True, random_state=42)`.
Scaling happens inside a `Pipeline`, so no information leaks between folds.

| Model | CV RMSE (log) |
|---|---|
| Dummy (predicts the mean) | 0.3993 |
| KNN (k=5, scaled) | 0.1935 |
| Random Forest (100 trees) | 0.1353 |
| Linear Regression | 0.1272 |
| **Ridge (alpha=100, scaled)** | **0.1198** |

Ridge's `alpha` was chosen with `GridSearchCV` over `[0.001 … 1000]`. The final model is
refit on the full training set, and predictions are converted back with `exp()`.

## Sources

- **Competition and data:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), Kaggle.
  Column definitions are in `data/raw/data_description.txt`.
- **Libraries:**
  [scikit-learn](https://scikit-learn.org/stable/),
  [pandas](https://pandas.pydata.org/docs/),
  [NumPy](https://numpy.org/doc/),
  [Matplotlib](https://matplotlib.org/stable/),
  [seaborn](https://seaborn.pydata.org/)
