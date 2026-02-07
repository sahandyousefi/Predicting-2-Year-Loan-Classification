# Model Card: Two-Year Delinquency Model

Structure follows Mitchell et al. (2019).

## Model details

- **Owner:** Sahand Yousefi
- **Version:** 0.1.0
- **Type:** binary classifier estimating the probability of default (PD), defined as 90 or more days past due within two years.
- **Candidates:** L1-regularized logistic regression on engineered features (champion), a WoE logistic scorecard, histogram gradient boosting, XGBoost and LightGBM. Selection uses `credit-risk compare` (ROC-AUC with DeLong tests, calibration and realized profit).
- **Code:** `src/credit_risk/models`. Configuration: `configs/default.yaml`.

## Intended use

- Ranking and pricing support for unsecured consumer credit, together with policy rules and analyst review.
- Generating principal reasons for adverse action notices from model attributions.
- **Out of scope:** fully automated decisions without human review, populations that differ materially from the training sample, and secured products where collateral drives loss.

## Factors

- **Age** is both an input and the protected attribute that is monitored. Regulation B permits age in an empirically derived, demonstrably and statistically sound scoring system, provided the age of an elderly applicant (62 or older) is not assigned a negative factor or value. `credit_risk.explain.pdp` and `credit_risk.explain.fairness` are used to check this.
- No other protected attribute (race, sex, national origin, marital status) is available, so proxy effects cannot be measured directly with this dataset.

## Metrics

| Metric | Purpose |
| --- | --- |
| ROC-AUC, Gini, KS | Rank ordering of applicants |
| PR-AUC | Performance on the default class |
| Brier score, ECE, Hosmer-Lemeshow | Calibration of PD estimates |
| Profit per applicant at the chosen cut-off | Business value under the cost assumptions in the config |
| Disparate impact ratio, equal opportunity difference | Approval outcomes by age band |

## Evaluation data

Stratified 80/20 split of the Credit Risk Benchmark Dataset (16,714 rows, `random_state=42`), with 5-fold cross-validation on the training part. Unit tests use synthetic data from `credit_risk.data.synthetic`.

## Quantitative analysis

Results from the exploration notebook (logistic regression with engineered features, held-out test set):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| 0.760 | 0.759 | 0.760 | 0.760 | 0.835 |

The notebook fitted the scaler before the train/test split; the package pipeline fixes this. Results of the package pipeline, the model comparison and the fairness report are to be filled from a full run on the benchmark data.

## Ethical considerations

- Reason codes describe the model, not the applicant's creditworthiness in general. They are grouped to borrower attributes so that notices never cite transformed variables.
- The benchmark has a far higher default rate than a production portfolio. PDs must be recalibrated to the target population's default rate before they are used for pricing or cut-offs.

## Caveats and recommendations

- The data has no application date, so out-of-time validation is not possible; production use requires it.
- Monitor input drift (PSI) and calibration monthly, and retrain or recalibrate when PSI exceeds 0.25 on key inputs.

## References

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., and Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 220-229.
- Consumer Financial Protection Bureau. Regulation B (Equal Credit Opportunity Act), 12 CFR Part 1002.
