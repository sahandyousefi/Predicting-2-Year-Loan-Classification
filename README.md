# Predicting-2-Year-Loan-Classification
Financial institutions must assess whether a borrower is at risk of becoming delinquent within the next 2 years.

Below is a **polished, professional, portfolio-ready GitHub README** for your project.
It incorporates the problem statement, dataset description, methodology, pipeline, and all interpretations **I1 → I9** that you asked me to memorize.

Copy-paste directly into your `README.md` file in your GitHub repo.

---

# **Credit Risk Prediction Using Logistic Regression & Feature Engineering**

A complete end-to-end machine learning project that predicts **2-year credit delinquency risk** using the *Credit Risk Benchmark Dataset*.
This project demonstrates expertise in:

✔️ Data cleaning & preprocessing
✔️ Exploratory Data Analysis (EDA)
✔️ Feature engineering
✔️ Outlier/log transformations
✔️ Multicollinearity handling
✔️ Class imbalance handling
✔️ Logistic Regression modeling
✔️ Hyperparameter tuning
✔️ Model interpretation (coefficients & feature impact)
✔️ Performance improvement & evaluation
✔️ Business insights

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

### ✔️ Summary from **I1**

* The target variable is slightly imbalanced.
* Delinquent borrowers tend to be **younger**.
* `debt_ratio`, `rev_util`, and `monthly_inc` contain **extreme outliers**.
* Late-payment features (`late_30_59`, `late_60_89`, `late_90`) are **~99% correlated**, causing severe multicollinearity.
* Some weak correlations found, indicating the need for feature engineering.

---

# **Feature–Target Correlation**

### Summary from **I2**

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

### Summary from **I3**

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

### Summary from **I4**

* Accuracy ≈ **0.66**
* Precision ≈ **0.68**
* Recall ≈ **0.60**
* F1 ≈ **0.64**
* ROC-AUC ≈ **0.73**

The model was stable but not strong enough for practical use.

---

# **Model Coefficients Interpretation**

### Summary from **I5**

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

# **Advanced Feature Engineering (Game Changer)**

To dramatically improve performance, new meaningful financial features were created:

1. **Debt per dependent**
2. **Income per dependent**
3. **Credit-to-income ratio**
4. **Late payment severity score** (weighted combination of all late-payment variables)
5. **Log transformations** to reduce skewness:

   * log_monthly_inc
   * log_rev_util
   * log_debt_ratio

This transformed the dataset into **highly predictive engineered features**.

---

# **Improved Logistic Regression Results (With Feature Engineering)**

### Summary from **I9**

After engineering 13 features:

| Metric    | Score     |
| --------- | --------- |
| Accuracy  | **0.760** |
| Precision | **0.759** |
| Recall    | **0.760** |
| F1 Score  | **0.760** |
| ROC-AUC   | **0.835** |

### 🚀 Improvement Highlights

* Accuracy improved **from 0.66 → 0.76**
* Recall improved **from 0.60 → 0.76**
* ROC-AUC improved **from 0.73 → 0.83**

This is a **massive improvement**, especially while keeping the model interpretable and business-friendly.

---

# **Tech Stack**

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook / Kaggle

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

Perfect for machine learning and data science job portfolios.

---

# **Next Steps (Future Work)**

To extend this project, consider:

* Random Forest vs. XGBoost comparison
* SHAP & model explainability
* Deployment using Flask/FastAPI
* Probability calibration
* Threshold optimization based on business cost

---

# ⭐ **If you like this project, consider giving the repo a star!**


