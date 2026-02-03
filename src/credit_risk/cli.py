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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credit-risk")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="fit the model and write metrics")
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None, help="overrides data.path")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=_train)

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
