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

- [ ] 2.6 Expected loss threshold selection (PD x LGD x EAD)
- [ ] 2.7 Model registry with XGBoost and LightGBM as optional extras
- [ ] 2.8 Optuna tuning with nested cross-validation
- [ ] 2.9 Model comparison report

## Phase 3: Explainability and fairness

- [ ] 3.1 SHAP global and local explanations
- [ ] 3.2 Adverse action reason codes from local attributions
- [ ] 3.3 Partial dependence and ICE curves
- [ ] 3.4 Age-band fairness analysis
- [ ] 3.5 Model card

## Phase 4: Monitoring and MLOps

- [ ] 4.1 PSI and CSI drift monitoring
- [ ] 4.2 Batch scoring command
- [ ] 4.3 Experiment tracking with MLflow and run fingerprints
- [ ] 4.4 Champion-challenger evaluation

## Phase 5: Serving

- [ ] 5.1 FastAPI scoring service
- [ ] 5.2 Dockerfile

## Phase 6: LLM applications

- [ ] 6.1 Provider-agnostic LLM client with a deterministic mock provider
- [ ] 6.2 Versioned prompt templates
- [ ] 6.3 Grounded adverse action notice generation
- [ ] 6.4 Credit policy corpus and retrieval with recall@k and MRR
- [ ] 6.5 Underwriter summary from score, reason codes and policy
- [ ] 6.6 LLM evaluation harness and prompt injection guardrails
- [ ] 6.7 Text embeddings of loan-purpose descriptions with an ablation study

## Phase 7: Documentation

- [ ] 7.1 docs/methodology.md
- [ ] 7.2 docs/llm_design.md with an architecture diagram
- [ ] 7.3 README rewrite for the package

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
