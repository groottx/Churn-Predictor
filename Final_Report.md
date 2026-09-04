# Final Report: Customer Churn Prediction
### Custom Logistic Regression (from scratch) — IBM Telco Customer Churn Dataset

**Author:** Aleena

---

## 1. Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Scratch LR (unweighted) | 0.800 | 0.643 | 0.554 | 0.595 | 0.841 |
| Scratch LR (class-weighted) | 0.738 | 0.504 | 0.783 | 0.614 | 0.841 |
| Scikit-learn LR (balanced) | 0.740 | 0.507 | 0.786 | 0.616 | 0.841 |

The class-weighted from-scratch model tracks scikit-learn's balanced logistic regression
almost exactly (ROC-AUC within 0.0004, recall within 0.003). This confirms the manual gradient
descent implementation — sigmoid, weighted log-loss, and the gradient update rule — is
mathematically correct, not just superficially similar.

**Why the weighted model has lower accuracy but is the better model for this problem:**
churn is imbalanced (~27% of customers). The unweighted model achieves higher accuracy
mainly by being conservative — predicting "no churn" more often — but it misses nearly half
of actual churners (55% recall). The class-weighted model catches 78% of churners, at the
cost of more false alarms. For a retention use case, missing a churner is more costly than
a false alarm (an unnecessary discount offer), so the weighted model is the right choice.

## 2. Key Insights

Ranked by standardized coefficient magnitude (impact on churn log-odds):

1. **Tenure** (coef ≈ −1.00) — the single strongest predictor. Newer customers are far more
   likely to churn; risk drops sharply the longer someone stays.
2. **Two-year contracts** (coef ≈ −0.62) — strongly protective against churn, consistent with
   the raw EDA (2.8% churn rate on two-year contracts vs. 42.7% on month-to-month).
3. **Fiber-optic internet** (coef ≈ +0.39) — associated with higher churn, likely reflecting
   price sensitivity or service dissatisfaction relative to DSL.
4. **Total charges** (coef ≈ +0.32) — higher lifetime spend correlates with higher churn risk
   in this data, worth investigating alongside contract type.
5. **One-year contracts** (coef ≈ −0.30) — also protective, though less than two-year.
6. **Electronic check payment** (coef ≈ +0.19) — correlates with higher churn, possibly a proxy
   for less "sticky" or less automated billing relationships.
7. **Online security & tech support subscriptions** (coef ≈ −0.19, −0.16) — customers with
   these add-ons churn less, suggesting engaged, higher-touch customers are stickier.

## 3. Business Implications & Recommendations

- **Target new, month-to-month, fiber customers first.** This segment combines the three
  strongest churn-risk factors (low tenure, no contract lock-in, fiber internet) and should
  be the priority for proactive retention outreach.
- **Incentivize contract upgrades.** Since two-year contracts are the strongest protective
  factor, offering a modest discount to move month-to-month customers onto annual contracts
  could meaningfully reduce churn.
- **Bundle sticky services.** Online security and tech support correlate with lower churn —
  offering free trials of these add-ons to at-risk customers may increase retention.
- **Adjust the decision threshold, not just the model.** Because a missed churner is costlier
  than a false alarm, a lower probability threshold (~0.3–0.4 rather than the default 0.5)
  is more aligned with the actual business cost tradeoff — this can be tuned without
  retraining, just by changing `decision_threshold` in the deployed artifact.

## 4. Limitations & Future Work

- Logistic regression assumes a linear relationship in log-odds space; it can't capture
  interaction effects (e.g., contract type × tenure) the way a tree-based model could.
- The dataset is a single static snapshot — a production system would need periodic
  retraining as pricing, competition, and customer behavior shift over time.
- Class-weighting was used to handle imbalance; SMOTE/oversampling is a worthwhile
  alternative to benchmark against in future iterations.
- The deployment demo (`predict_churn()`) is a scoring function, not a hosted API — a next
  step would be wrapping it in a lightweight Flask/FastAPI service for real integration
  with a CRM.

## 5. Deliverables Summary

| Deliverable | File |
|---|---|
| Data loading & cleaning | `src/data_prep.py` |
| From-scratch logistic regression | `src/model.py` |
| Training (weighted vs. unweighted) | `src/train.py` |
| Evaluation & scikit-learn benchmark | `src/evaluate.py` |
| Deployment scoring function & artifact | `src/deploy.py` |
| Full pipeline runner | `main.py` |
| Saved model artifact | `churn_model_artifact.json` |
| Diagnostic plots | `confusion_matrices.png`, `roc_curves.png`, `feature_importance.png` |
