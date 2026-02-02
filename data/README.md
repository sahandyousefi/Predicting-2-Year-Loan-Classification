# Data

## Source

The project uses the **Credit Risk Benchmark Dataset** published on Kaggle. It contains 16,714 borrowers, ten numeric attributes and the binary target `dlq_2yrs`. The raw file is not committed to this repository.

## Download

1. Search for "Credit Risk Benchmark Dataset" on Kaggle and download the CSV, or use the Kaggle CLI with the dataset's identifier.
2. Place the file at `data/Credit Risk Benchmark Dataset.csv`, or point `data.path` in `configs/default.yaml` to its location.

## Data dictionary

| Column | Description |
| --- | --- |
| `rev_util` | Revolving utilization: balance on credit cards and personal lines divided by their limits |
| `age` | Borrower age in years |
| `late_30_59` | Number of times 30 to 59 days past due in the last two years |
| `debt_ratio` | Monthly debt payments and living costs divided by monthly gross income |
| `monthly_inc` | Monthly gross income |
| `open_credit` | Number of open loans and lines of credit |
| `late_90` | Number of times 90 or more days past due |
| `real_estate` | Number of mortgage and real estate loans |
| `late_60_89` | Number of times 60 to 89 days past due in the last two years |
| `dependents` | Number of dependents, excluding the borrower |
| `dlq_2yrs` | Target: 1 if the borrower was 90 or more days past due within two years |

## Validation

`credit_risk.data.io.load_dataset` checks every file against `credit_risk.data.schema`:

- Errors stop the run: missing columns, non-numeric types, negative counts or amounts, and a target that is not coded 0/1.
- Warnings are logged only: values outside plausible ranges (for example age above 100 or utilization above 10) and duplicated rows.

Tests and CI never read this file. They use `credit_risk.data.synthetic.make_synthetic`, which produces data with the same schema.
