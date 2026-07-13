# Kaggle

Tabular ML projects for Kaggle competitions. Each folder is a self-contained competition.

## Folder structure

```
kaggle/
├── house_price/     # House Prices — regression
├── titanic/         # Titanic — classification
└── README.md
```

---

## `house_price/` — [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Predict sale price from house features.

| File / folder | What it is |
|---|---|
| `house_price.ipynb` | Main notebook — EDA, preprocessing, model training, evaluation |
| `submission.csv` | Predictions to upload to Kaggle |
| `data/` | Raw competition data |
| `data/train.csv` | Training set (features + SalePrice) |
| `data/test.csv` | Test set (features only — predict these) |
| `data/sample_submission.csv` | Kaggle's required submission format |
| `data/data_description.txt` | Feature definitions from Kaggle |
| `data_pre_processing.txt` | Notes on cleaning and preprocessing steps |
| `results.md` | Model comparison scores and takeaways |
| `xgboost_votingRegressor_fix.text` | Troubleshooting notes for XGBoost / VotingRegressor errors |
| `catboost_info/` | Auto-generated CatBoost training logs (safe to ignore) |

---

## `titanic/` — [Titanic](https://www.kaggle.com/c/titanic)

Predict passenger survival (0 or 1).

| File / folder | What it is |
|---|---|
| `titanic.ipynb` | Main notebook — EDA, preprocessing, model training, evaluation |
| `submission.csv` | Predictions to upload to Kaggle |
| `data/` | Raw competition data |
| `data/train.csv` | Training set (features + Survived) |
| `data/test.csv` | Test set (features only — predict these) |
| `data/sample_submission.csv` | Kaggle's required submission format |
| `data_pre_processing.txt` | Notes on cleaning and preprocessing steps |
| `optimization.txt` | Ideas to improve score (feature engineering, models, tuning) |

---

## Typical workflow

1. Open the `.ipynb` notebook and run through EDA → cleaning → training
2. Check `data_pre_processing.txt` for a quick summary of preprocessing decisions
3. Upload `submission.csv` to Kaggle
4. Use `optimization.txt` / `results.md` for ideas to improve score
