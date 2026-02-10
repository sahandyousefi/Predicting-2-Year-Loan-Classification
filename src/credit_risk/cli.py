from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence

from credit_risk.config import load_config


def _train(args: argparse.Namespace) -> int:
    from credit_risk.models.train import run

    cfg = load_config(args.config)
    report = run(args.data, args.output_dir, config=cfg)
    print(json.dumps(report, indent=2))
    return 0


def _validate(args: argparse.Namespace) -> int:
    import pandas as pd

    from credit_risk.data.schema import validate_frame

    df = pd.read_csv(args.data)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    report = validate_frame(df, require_target=not args.no_target)
    for line in report.errors:
        print(f"ERROR   {line}")
    for line in report.warnings:
        print(f"WARNING {line}")
    print(
        f"{len(df)} rows, {len(report.errors)} errors, {len(report.warnings)} warnings"
    )
    return 0 if report.ok else 1


def _synthesize(args: argparse.Namespace) -> int:
    from credit_risk.data.synthetic import SyntheticSpec, make_synthetic

    df = make_synthetic(
        SyntheticSpec(n=args.n, base_rate=args.base_rate, seed=args.seed)
    )
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    from pathlib import Path

    from sklearn.model_selection import train_test_split

    from credit_risk.data.io import load_dataset
    from credit_risk.evaluation.compare import (
        compare_models,
        compare_models_with_intervals,
        to_markdown,
    )
    from credit_risk.features.engineering import build_xy

    cfg = load_config(args.config)
    X, y = build_xy(load_dataset(args.data or cfg.data.path))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        stratify=y,
        random_state=cfg.data.random_state,
    )
    out = Path(args.output_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.n_boot > 0:
        table, intervals = compare_models_with_intervals(
            args.models,
            X_tr,
            y_tr,
            X_te,
            y_te,
            cfg.cost,
            monotone=args.monotone,
            n_boot=args.n_boot,
            seed=cfg.data.random_state,
        )
        intervals.to_csv(out / "model_comparison_ci.csv", index=False)
        print(to_markdown(intervals.round(4)) + "\n")
    else:
        table = compare_models(
            args.models, X_tr, y_tr, X_te, y_te, cfg.cost, monotone=args.monotone
        )
    table.to_csv(out / "model_comparison.csv", index=False)
    (out / "model_comparison.md").write_text(to_markdown(table) + "\n")
    print(to_markdown(table))
    return 0


def _drift(args: argparse.Namespace) -> int:
    from credit_risk.data.io import load_dataset
    from credit_risk.data.schema import FEATURES
    from credit_risk.monitoring.drift import drift_report

    ref = load_dataset(args.reference, require_target=False)
    cur = load_dataset(args.current, require_target=False)
    report = drift_report(ref, cur, FEATURES)
    print(report.to_string(index=False))
    if args.output:
        report.to_csv(args.output, index=False)
    return 1 if (report["status"] == "significant").any() and args.fail_on_drift else 0


def _score(args: argparse.Namespace) -> int:
    from credit_risk.data.io import load_dataset
    from credit_risk.models.persistence import load_bundle
    from credit_risk.serving.batch import score_frame

    bundle = load_bundle(args.model)
    raw = load_dataset(args.input, require_target=False)
    scored = score_frame(bundle, raw)
    if args.target_rate is not None:
        from credit_risk.models.prior_shift import adjust_to_prior

        scored["pd_target"] = adjust_to_prior(
            scored["pd"], args.source_rate, args.target_rate
        ).round(6)
    scored.to_csv(args.output, index=False)
    share = (scored["decision"] == "approve").mean()
    print(f"scored {len(scored)} rows, approval rate {share:.3f}, wrote {args.output}")
    return 0


def _challenge(args: argparse.Namespace) -> int:
    from pathlib import Path

    from credit_risk.data.io import load_dataset
    from credit_risk.data.schema import TARGET
    from credit_risk.models.persistence import load_bundle
    from credit_risk.monitoring.champion_challenger import (
        PromotionRule,
        challenge_bundles,
        report_markdown,
    )

    cfg = load_config(args.config)
    df = load_dataset(args.data)
    result = challenge_bundles(
        load_bundle(args.champion),
        load_bundle(args.challenger),
        df.drop(columns=[TARGET]),
        df[TARGET],
        cfg.cost,
        PromotionRule(auc_margin=args.auc_margin, n_boot=args.n_boot),
    )
    text = report_markdown(result)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0


def _text_ablation(args: argparse.Namespace) -> int:
    from credit_risk.data.io import load_dataset
    from credit_risk.data.synthetic import SyntheticSpec, make_synthetic
    from credit_risk.evaluation.ablation import signal_sweep
    from credit_risk.evaluation.compare import to_markdown

    df = (
        load_dataset(args.data)
        if args.data
        else make_synthetic(SyntheticSpec(n=args.n, seed=args.seed))
    )
    table = signal_sweep(
        df,
        signals=args.signals,
        seed=args.seed,
        learners=args.learners,
        n_components=args.dims,
        backend=args.backend,
        n_splits=args.folds,
        n_repeats=args.repeats,
    ).round(4)
    if args.output:
        table.to_csv(args.output, index=False)
    print(to_markdown(table))
    return 0


def _oot(args: argparse.Namespace) -> int:
    from pathlib import Path

    from credit_risk.data.io import load_dataset
    from credit_risk.data.vintage import VINTAGE_COLUMN, add_vintage
    from credit_risk.evaluation.compare import to_markdown
    from credit_risk.evaluation.oot import oot_validation

    df = load_dataset(args.data)
    if VINTAGE_COLUMN not in df.columns:
        print(f"no '{VINTAGE_COLUMN}' column, assigning synthetic vintages")
        df = add_vintage(df, args.start, args.n_months, args.shift, seed=args.seed)
    report = oot_validation(df, args.cutoff, args.model, seed=args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.summary.to_csv(out / "oot_summary.csv", index=False)
    report.by_vintage.to_csv(out / "oot_by_vintage.csv", index=False)
    print(to_markdown(report.summary.round(4)))
    stability = report.score_stability
    print(f"score PSI {stability['psi']} ({stability['status']})")
    return 0


def _conformal(args: argparse.Namespace) -> int:
    from sklearn.model_selection import train_test_split

    from credit_risk.data.io import load_dataset
    from credit_risk.data.schema import TARGET
    from credit_risk.evaluation.compare import to_markdown
    from credit_risk.features.engineering import add_engineered_features
    from credit_risk.models.conformal import uncertainty_report
    from credit_risk.models.persistence import load_bundle

    bundle = load_bundle(args.model)
    df = load_dataset(args.data)
    cal, test = train_test_split(
        df, train_size=args.cal_size, stratify=df[TARGET], random_state=args.seed
    )

    def score(frame):
        X = add_engineered_features(frame)[bundle.features]
        return bundle.model.predict_proba(X)[:, 1]

    sets, va = uncertainty_report(
        cal[TARGET], score(cal), test[TARGET], score(test), tuple(args.alpha)
    )
    print(to_markdown(sets.round(4)))
    print(" ".join(f"{k}={v:.4f}" for k, v in va.items()))
    return 0


def _reject_inference(args: argparse.Namespace) -> int:
    from credit_risk.data.through_the_door import (
        ThroughTheDoorSpec,
        make_through_the_door,
    )
    from credit_risk.evaluation.compare import to_markdown
    from credit_risk.models.reject_inference import reject_inference_study

    spec = ThroughTheDoorSpec(
        n=args.n,
        base_rate=args.base_rate,
        approval_rate=args.approval_rate,
        hidden_signal=args.hidden_signal,
        seed=args.seed,
    )
    table = reject_inference_study(
        make_through_the_door(spec), inflation=args.inflation, seed=args.seed
    ).round(4)
    if args.output:
        table.to_csv(args.output, index=False)
    print(to_markdown(table))
    return 0


def _survival(args: argparse.Namespace) -> int:
    from credit_risk.data.survival import SurvivalSpec, simulate_survival
    from credit_risk.evaluation.compare import to_markdown
    from credit_risk.models.survival import survival_study

    sample = simulate_survival(
        SurvivalSpec(
            n=args.n, pd_24=args.pd_24, censor_rate=args.censor_rate, seed=args.seed
        )
    )
    true_pd = {h: sample.true_cumulative_pd(h) for h in args.horizons}
    table = survival_study(
        sample.frame, true_pd, horizon=24 if 24 in true_pd else args.horizons[0]
    ).round(4)
    if args.output:
        table.to_csv(args.output, index=False)
    print(to_markdown(table))
    return 0


def _extract_eval(args: argparse.Namespace) -> int:
    from credit_risk.data.synthetic_applications import make_applications
    from credit_risk.evaluation.compare import to_markdown
    from credit_risk.llm.client import MockLLM, get_client
    from credit_risk.llm.evaluation import evaluate_extraction
    from credit_risk.llm.extraction_baseline import rule_based_responder

    sample = make_applications(
        args.n, args.missing_rate, args.hard_rate, seed=args.seed
    )
    if args.provider == "rules":
        client = MockLLM(rule_based_responder, model="rule-based")
    else:
        client = get_client(args.provider)
    table, summary = evaluate_extraction(client, sample.texts, sample.gold)
    if args.output:
        table.to_csv(args.output, index=False)
    print(to_markdown(table.round(3)))
    usage = summary.pop("usage")
    print(" ".join(f"{k}={v}" for k, v in summary.items()))
    print(" ".join(f"{k}={v}" for k, v in usage.items()))
    return 0


def _consistency(args: argparse.Namespace) -> int:
    from credit_risk.data.synthetic_applications import make_applications
    from credit_risk.evaluation.compare import to_markdown
    from credit_risk.llm.client import MockLLM, get_client
    from credit_risk.llm.consistency import misstatement_study, reconcile
    from credit_risk.llm.extraction import extract_fields
    from credit_risk.llm.extraction_baseline import rule_based_responder

    sample = make_applications(
        args.n,
        args.missing_rate,
        args.hard_rate,
        seed=args.seed,
        misstatement_rate=args.misstatement_rate,
    )
    if args.provider == "rules":
        client = MockLLM(rule_based_responder, model="rule-based")
    else:
        client = get_client(args.provider)
    reports = [
        reconcile(extract_fields(text, client).values, bureau)
        for text, (_, bureau) in zip(sample.texts, sample.attributes.iterrows())
    ]
    table = misstatement_study(reports, sample.misstated).round(3)
    if args.output:
        table.to_csv(args.output, index=False)
    print(to_markdown(table))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credit-risk")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="fit the model and write metrics")
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None, help="overrides data.path")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=_train)

    p = sub.add_parser("compare", help="compare models on one split")
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "scorecard", "hist_gbm", "xgboost", "lightgbm"],
    )
    p.add_argument(
        "--monotone", action="store_true", help="monotonic constraints for GBMs"
    )
    p.add_argument(
        "--n-boot", type=int, default=0, help="bootstrap intervals for all metrics"
    )
    p.set_defaults(func=_compare)

    p = sub.add_parser("drift", help="PSI of each input between two samples")
    p.add_argument("reference")
    p.add_argument("current")
    p.add_argument("--output", default=None)
    p.add_argument("--fail-on-drift", action="store_true")
    p.set_defaults(func=_drift)

    p = sub.add_parser("score", help="batch-score a CSV with a saved model")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument(
        "--target-rate", type=float, default=None, help="portfolio default rate"
    )
    p.add_argument(
        "--source-rate",
        type=float,
        default=0.5,
        help="rate the PDs are calibrated to; 0.5 for balanced class weights",
    )
    p.set_defaults(func=_score)

    p = sub.add_parser("challenge", help="champion-challenger comparison")
    p.add_argument("data", help="labeled out-of-sample CSV")
    p.add_argument("--champion", required=True)
    p.add_argument("--challenger", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--auc-margin", type=float, default=0.005)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--output", default=None, help="write the Markdown report")
    p.set_defaults(func=_challenge)

    p = sub.add_parser("text-ablation", help="ablation of text embedding features")
    p.add_argument("--data", default=None, help="labeled CSV; synthetic if omitted")
    p.add_argument("--n", type=int, default=5000, help="synthetic sample size")
    p.add_argument("--signals", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    p.add_argument("--learners", nargs="+", default=["logistic", "hist_gbm"])
    p.add_argument("--dims", type=int, nargs="+", default=[8, 16])
    p.add_argument("--backend", choices=["tfidf", "sbert"], default="tfidf")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    p.set_defaults(func=_text_ablation)

    p = sub.add_parser("oot", help="out-of-time validation by application month")
    p.add_argument("data")
    p.add_argument("--cutoff", required=True, help="first out-of-time month YYYY-MM")
    p.add_argument("--model", default="logistic")
    p.add_argument("--start", default="2019-01", help="first synthetic vintage")
    p.add_argument("--n-months", type=int, default=24)
    p.add_argument("--shift", type=float, default=0.0, help="synthetic drift")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="artifacts")
    p.set_defaults(func=_oot)

    p = sub.add_parser("conformal", help="conformal sets and Venn-Abers intervals")
    p.add_argument("data", help="labeled CSV not used for training")
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--cal-size", type=float, default=0.5)
    p.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_conformal)

    p = sub.add_parser(
        "reject-inference", help="reject inference study on synthetic applicants"
    )
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--base-rate", type=float, default=0.12)
    p.add_argument("--approval-rate", type=float, default=0.6)
    p.add_argument(
        "--hidden-signal",
        type=float,
        default=0.25,
        help="outcome information used by the old policy and absent from the data",
    )
    p.add_argument("--inflation", type=float, default=1.5, help="parceling factor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    p.set_defaults(func=_reject_inference)

    p = sub.add_parser(
        "survival", help="time-to-default model on synthetic performance data"
    )
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--pd-24", type=float, default=0.15)
    p.add_argument(
        "--censor-rate",
        type=float,
        default=0.03,
        help="monthly probability of leaving observation without default",
    )
    p.add_argument("--horizons", type=int, nargs="+", default=[12, 24, 36])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    p.set_defaults(func=_survival)

    p = sub.add_parser(
        "extract-eval", help="field extraction accuracy on synthetic applications"
    )
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--missing-rate", type=float, default=0.15)
    p.add_argument(
        "--hard-rate",
        type=float,
        default=0.3,
        help="share of fields stated with phrasings the rule baseline does not cover",
    )
    p.add_argument(
        "--provider",
        default="rules",
        help="'rules' for the rule-based baseline, otherwise an LLM provider",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    p.set_defaults(func=_extract_eval)

    p = sub.add_parser("consistency", help="detection of misstated application fields")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--misstatement-rate", type=float, default=0.3)
    p.add_argument("--missing-rate", type=float, default=0.15)
    p.add_argument("--hard-rate", type=float, default=0.3)
    p.add_argument("--provider", default="rules")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    p.set_defaults(func=_consistency)

    p = sub.add_parser("validate", help="check a CSV against the schema")
    p.add_argument("data")
    p.add_argument("--no-target", action="store_true", help="scoring file")
    p.set_defaults(func=_validate)

    p = sub.add_parser("synthesize", help="write a synthetic dataset")
    p.add_argument("output")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--base-rate", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_synthesize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
