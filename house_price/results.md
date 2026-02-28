# House Prices

## Model Results (from `note.ipynb`)

The notebook currently uses 9 models. Values below are copied from executed notebook outputs.

| # | Model | Metric shown in notebook | Result |
|---|---|---|---|
| 1 | Linear Regression | Test MSE (`mean_squared_error`) | `1.5106448817240645e+18` |
| 2 | Random Forest Regressor | CV RMSE (`sqrt(-best_score_)`) | `0.13383339873228997` |
| 3 | XGBoost Regressor | CV RMSE (`sqrt(-best_score_)`) | `0.11871798784728256` |
| 4 | Ridge Regression | CV RMSE (`sqrt(-best_score_)`) | `0.10906292948679212` |
| 5 | Gradient Boosting Regressor | CV RMSE (`sqrt(-best_score_)`) | `0.11327331247916332` |
| 6 | LightGBM Regressor | CV RMSE (`sqrt(-best_score_)`) | `0.12740886862621015` |
| 7 | CatBoost Regressor | CV RMSE (`sqrt(-best_score_)`) | `0.11475569741494256` |
| 8 | Voting Regressor | Test RMSE (`mean_squared_error(..., squared=False)`) | `0.11934429677311026` |
| 9 | Stacking Regressor | Test RMSE (`mean_squared_error(..., squared=False)`) | `0.11893874315302513` |

## Summary and Opinion

- Best single-model CV result is **Ridge Regression** (`0.10906` CV RMSE).
- Best ensemble holdout result is **Stacking Regressor** (`0.11894` test RMSE), slightly better than Voting.
- The **Linear Regression** result is unusually large (very high MSE), which suggests a target/prediction scale mismatch for that model setup.
- If choosing one model for final submission from the current notebook, I would pick **Stacking Regressor**, since it combines strong models and is validated on holdout RMSE.

## Improvements for Future Iterations

- Evaluate all 9 models on the **same metric and same split** (for example, RMSE on validation fold) so comparisons are strictly fair.
- Use **K-fold cross-validation** for the final ensemble as well, not only for base models.
- Add a tracking table with: model name, params, CV RMSE, holdout RMSE, and Kaggle public score.
- Re-check target transform consistency (`log` vs `exp`) for each model path to avoid scale mismatch like the linear regression result.
- Tune stacking meta-model explicitly (final estimator type/params) and test weighted blending between top models.