import pytest

from credit_risk.config import Config, load_config


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.data.target == "dlq_2yrs"
    assert 0 < cfg.cost.lgd <= 1


def test_partial_config_uses_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  C: 0.5\n")
    cfg = load_config(p)
    assert cfg.model.C == 0.5
    assert cfg.model.cv_folds == 5


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  depth: 3\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_invalid_value_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  test_size: 1.5\n")
    with pytest.raises(ValueError):
        load_config(p)
