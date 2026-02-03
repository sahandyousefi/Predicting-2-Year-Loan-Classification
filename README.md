# Credit Risk Prediction: 2-Year Delinquency Classification

A complete end-to-end machine learning project that predicts **2-year credit delinquency risk** using the *Credit Risk Benchmark Dataset*.
The project covers:

* Data cleaning & preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Outlier/log transformations
* Multicollinearity handling
* Class imbalance handling
* Logistic Regression modeling
* Hyperparameter tuning
* Model interpretation (coefficients & feature impact)
* Performance improvement & evaluation
* Business insights

This project is built entirely using **Python**, **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, and **scikit-learn**, and is designed for both technical and non-technical audiences.

---

# **Problem Statement**

Financial institutions must assess whether a borrower is at risk of becoming **delinquent within the next 2 years**.
The goal of this project is to build a **predictive model** that estimates this probability using borrower attributes such as:

* Age
* Income
* Debt ratio
* Credit utilization
* Credit history
* Late payment patterns
* Number of dependents

This model helps institutions **manage credit risk**, improve approval decisions, and reduce financial exposure.

---

# **Dataset Overview**

The dataset contains numerical and financial attributes describing borrower behavior.
The target variable is:

* `dlq_2yrs` → 1 = borrower became delinquent within 2 years, 0 = did not

Dataset contains:

* No missing values
* Mostly continuous features
* Skewed financial variables
* Highly correlated late-payment features

---

# **EDA Summary**

* The target variable is slightly imbalanced.
* Delinquent borrowers tend to be **younger**.
* `debt_ratio`, `rev_util`, and `monthly_inc` contain **extreme outliers**.
* Late-payment features (`late_30_59`, `late_60_89`, `late_90`) are **~99% correlated**, causing severe multicollinearity.
* Some weak correlations found, indicating the need for feature engineering.

---

# **Feature–Target Correlation**


Most correlated positively with delinquency:

* `late_30_59`
* `late_60_89`
* `late_90`
* `dependents`

Most negatively correlated:

* `age`
* `monthly_inc`
* `open_credit`

Younger, lower-income individuals with late payments are more likely to default.

---

# **Feature Scaling Strategy**

Logistic Regression is sensitive to feature magnitude.
We scaled key continuous variables (age, income, debt, utilization, dependents) using **StandardScaler**.

Scaling improved:

* gradient descent convergence
* model stability
* coefficient interpretability
* overall predictive performance

---

# **Baseline Logistic Regression Performance**

Before feature engineering:

* Accuracy ≈ **0.66**
* Precision ≈ **0.68**
* Recall ≈ **0.60**
* F1 ≈ **0.64**
* ROC-AUC ≈ **0.73**

The model was stable but not strong enough for practical use.

---

# **Model Coefficients Interpretation**

* Strongest positive predictor: **late_90**
* Strongest negative predictor: **age**
* Income reduces default risk
* Dependents slightly increase risk
* Credit exposure has moderate effect

These align with real-world financial logic.

---

# **Hyperparameter Tuning**

GridSearchCV was used to tune:

* `penalty` (L1/L2)
* `C`
* `class_weight`

Performance did **not** improve significantly — showing that the model had reached its linear performance limit.

---

# **Feature Engineering**

The following financial features were derived from the raw attributes:

1. **Debt per dependent**
2. **Income per dependent**
3. **Credit-to-income ratio**
4. **Late payment severity score** (weighted combination of all late-payment variables)
5. **Log transformations** to reduce skewness:

   * log_monthly_inc
   * log_rev_util
   * log_debt_ratio


---

# **Improved Logistic Regression Results (With Feature Engineering)**

After engineering 13 features:

| Metric    | Score     |
| --------- | --------- |
| Accuracy  | **0.760** |
| Precision | **0.759** |
| Recall    | **0.760** |
| F1 Score  | **0.760** |
| ROC-AUC   | **0.835** |

### Improvement Highlights

* Accuracy improved **from 0.66 → 0.76**
* Recall improved **from 0.60 → 0.76**
* ROC-AUC improved **from 0.73 → 0.83**

The gain comes from feature engineering alone, so the model remains a fully interpretable linear classifier.

---

# **Project Structure**

```
.
├── notebooks/01_exploration_and_baseline.ipynb   EDA, experiments and interpretation
├── src/credit_risk/
│   ├── features/engineering.py                   feature engineering and input validation
│   └── models/train.py                           training pipeline, CV, threshold tuning
├── tests/
│   └── test_pipeline.py                          unit tests on synthetic data
└── pyproject.toml
```

The `credit_risk` package turns the final notebook model into a reproducible pipeline. Scaling is fit inside an sklearn `Pipeline`, so it only sees training data, and the decision threshold is tuned on the training split rather than the test set.

---

# **How to Run**

```bash
pip install -e ".[dev]"

# Download the Credit Risk Benchmark Dataset from Kaggle into data/
python -m credit_risk.models.train --data "data/Credit Risk Benchmark Dataset.csv"

pytest -q
```

Training writes cross-validated ROC-AUC, test metrics at the default and tuned thresholds, and model coefficients to `artifacts/metrics.json`.

---

# **Tech Stack**

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook / Kaggle
* pytest

---

# **Conclusion**

This project demonstrates how:

* Proper **EDA**
* Careful **handling of multicollinearity**
* Smart **feature engineering**
* Scaling & transformations
* Logistic Regression tuning

can elevate a basic model into a **high-performing, interpretable, and business-ready predictive system**.

The final model delivers:

* Strong predictive power
* Real-world financial interpretability
* Solid performance metrics
* Clean and well-documented ML workflow

---

# **Next Steps (Future Work)**

To extend this project, consider:

* Random Forest vs. XGBoost comparison
* SHAP & model explainability
* Deployment using Flask/FastAPI
* Probability calibration
* Threshold optimization based on business cost



