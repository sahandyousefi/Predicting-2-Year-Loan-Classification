# Roadmap

This file tracks the development plan for the credit risk project. The notebook remains the record of the original exploration; the package under `src/credit_risk` is the production version of that work. Items are numbered `<phase>.<item>` and move to the Completed section when they land.

## Target layout

```
.
├── configs/default.yaml          model, split and cost parameters
├── data/README.md                dataset source and download steps
├── docs/                         methodology, LLM design, model card
├── notebooks/                    exploration notebook
├── src/credit_risk/
│   ├── config.py                 typed configuration
│   ├── cli.py                    command line entry points
│   ├── data/                     schema, validation, loading, synthetic data
│   ├── features/                 engineered features, WoE binning
│   ├── models/                   pipelines, scorecard, registry, tuning
│   ├── evaluation/               metrics, calibration, DeLong, thresholds
│   ├── explain/                  SHAP, reason codes, PDP/ICE, fairness
│   ├── monitoring/               drift, experiment tracking
│   ├── serving/                  FastAPI service, batch scoring
│   └── llm/                      LLM client, prompts, retrieval, notices
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
├── Makefile
└── Dockerfile
```

## Phase 1: Foundations


## Phase 2: Modeling


## Phase 3: Explainability and fairness


## Phase 4: Monitoring and MLOps


## Phase 5: Serving


## Phase 6: LLM applications


## Phase 7: Documentation


## Phase 8: Validation and robustness


## Phase 9: Advanced modeling


## Phase 10: LLM applications II


## Phase 11: Reporting

- [ ] 11.1 Notebooks that reproduce the key figures on synthetic data
- [ ] 11.2 Static HTML validation report generated from a run
- [ ] 11.3 Streamlit app for single-applicant scoring and explanation

## Completed

- [x] 2026-02-02: 1.1 Package layout under src/credit_risk with pyproject.toml and optional extras
- [x] 2026-02-02: 1.2 Replace the deprecated logistic regression penalty argument
- [x] 2026-02-02: 1.3 YAML configuration with typed dataclasses
- [x] 2026-02-02: 1.4 Data schema and validation checks
- [x] 2026-02-02: 1.5 Synthetic data generator matching the benchmark schema
- [x] 2026-02-02: 1.6 Dataset loader with validation and data/README.md
- [x] 2026-02-03: 1.7 Ruff and pre-commit configuration
- [x] 2026-02-03: 1.8 GitHub Actions CI on Python 3.10 and 3.12
- [x] 2026-02-03: 1.9 Makefile for common tasks
- [x] 2026-02-03: 1.10 MIT license
- [x] 2026-02-03: 1.11 Move the exploration notebook to notebooks/
- [x] 2026-02-03: 1.12 Command line interface
- [x] 2026-02-03: 2.1 Weight of Evidence binning and Information Value
- [x] 2026-02-04: 2.2 Logistic scorecard with points-to-double-odds scaling
- [x] 2026-02-04: 2.3 Credit metrics: KS, Gini, lift and gain tables
- [x] 2026-02-04: 2.4 Probability calibration with Brier score and reliability curves
- [x] 2026-02-04: 2.5 DeLong test for paired ROC-AUC comparison
- [x] 2026-02-04: 2.6 Expected loss threshold selection (PD x LGD x EAD)
- [x] 2026-02-05: 2.7 Model registry with XGBoost and LightGBM as optional extras
- [x] 2026-02-05: 2.8 Optuna tuning with nested cross-validation
- [x] 2026-02-05: 2.9 Model comparison report
- [x] 2026-02-05: 3.1 SHAP global and local explanations
- [x] 2026-02-05: 3.2 Adverse action reason codes from local attributions
- [x] 2026-02-05: 3.3 Partial dependence and ICE curves
- [x] 2026-02-05: 3.4 Age-band fairness analysis
- [x] 2026-02-05: 3.5 Model card
- [x] 2026-02-06: 4.1 PSI and CSI drift monitoring
- [x] 2026-02-06: 4.2 Batch scoring command
- [x] 2026-02-06: 4.3 Experiment tracking with MLflow and run fingerprints
- [x] 2026-02-06: 5.1 FastAPI scoring service
- [x] 2026-02-06: 5.2 Dockerfile
- [x] 2026-02-07: 6.1 Provider-agnostic LLM client with a deterministic mock provider
- [x] 2026-02-07: 6.2 Versioned prompt templates
- [x] 2026-02-07: 6.3 Grounded adverse action notice generation
- [x] 2026-02-07: 6.4 Credit policy corpus and retrieval with recall@k and MRR
- [x] 2026-02-07: 6.5 Underwriter summary from score, reason codes and policy
- [x] 2026-02-07: 6.6 LLM evaluation harness and prompt injection guardrails
- [x] 2026-02-07: 7.3 README rewrite for the package
- [x] 2026-02-08: 4.4 Champion-challenger evaluation
- [x] 2026-02-08: 6.7a Synthetic loan-purpose descriptions and a text embedding module
- [x] 2026-02-08: 6.7b Ablation study of text embedding features
- [x] 2026-02-08: 7.1 docs/methodology.md
- [x] 2026-02-08: 7.2 docs/llm_design.md with an architecture diagram
- [x] 2026-02-08: 8.1 Synthetic vintage column and out-of-time validation
- [x] 2026-02-09: 8.2 Monotonic constraints in gradient boosting models
- [x] 2026-02-09: 8.3 Bootstrap confidence intervals for every reported metric
- [x] 2026-02-09: 8.4 Conformal prediction and Venn-Abers intervals for PD
- [x] 2026-02-09: 8.5 Prior-shift recalibration of PD to a target portfolio default rate
- [x] 2026-02-09: 8.6 Reject inference on a synthetic through-the-door population
- [x] 2026-02-10: 9.1 Discrete-time survival model for time to default
- [x] 2026-02-10: 9.2 Tabular neural network baseline in PyTorch as an optional extra
- [x] 2026-02-10: 9.3 Stacked ensemble with out-of-fold predictions
- [x] 2026-02-10: 10.1a Extraction schema, quote-grounded validation and rule-based baseline
- [x] 2026-02-10: 10.1b Synthetic application texts and field-level extraction evaluation
- [x] 2026-02-10: 10.2 Consistency checks between extracted fields and bureau attributes
- [x] 2026-02-10: 10.3 Prompt version comparison report on the evaluation sets
