import pandas as pd

from credit_risk.cli import main
from credit_risk.data.schema import TARGET
from credit_risk.models.persistence import load_bundle
from credit_risk.serving.batch import score_frame


def test_train_then_batch_score(tmp_path):
    data = tmp_path / "s.csv"
    main(["synthesize", str(data), "--n", "800"])
    out_dir = tmp_path / "art"
    main(["train", "--data", str(data), "--output-dir", str(out_dir)])
    bundle = load_bundle(out_dir / "model.joblib")

    raw = pd.read_csv(data).drop(columns=[TARGET]).head(50)
    scored = score_frame(bundle, raw)
    assert scored["pd"].between(0, 1).all()
    declined = scored[scored["decision"] == "decline"]
    assert (declined["reason_codes"].str.len() > 0).all()
    assert (scored.loc[scored["decision"] == "approve", "reason_codes"] == "").all()

    inp, outp = tmp_path / "in.csv", tmp_path / "out.csv"
    raw.to_csv(inp, index=False)
    assert (
        main(["score", str(inp), str(outp), "--model", str(out_dir / "model.joblib")])
        == 0
    )
    assert len(pd.read_csv(outp)) == 50

    shifted = tmp_path / "shifted.csv"
    args = ["score", str(inp), str(shifted), "--model", str(out_dir / "model.joblib")]
    assert main([*args, "--target-rate", "0.05"]) == 0
    result = pd.read_csv(shifted)
    assert (result["pd_target"] <= result["pd"] + 1e-6).all()
    assert (result["decision"] == pd.read_csv(outp)["decision"]).all()
