import pytest

from credit_risk.llm.client import MockLLM, RetryingClient, UsageLog, get_client


def test_mock_is_deterministic():
    llm = MockLLM()
    a = llm.complete("sys", "hello").text
    b = llm.complete("sys", "hello").text
    assert a == b and a != llm.complete("sys", "other").text
    assert len(llm.calls) == 3


def test_retry_then_succeed():
    attempts = {"n": 0}

    def flaky(system, user):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TimeoutError("slow")
        return "ok"

    sleeps = []
    client = RetryingClient(MockLLM(flaky), max_retries=3, sleep=sleeps.append)
    assert client.complete("s", "u").text == "ok"
    assert sleeps == [1.0, 2.0]
    assert client.usage.summary()["calls"] == 1


def test_retry_gives_up():
    def broken(system, user):
        raise ConnectionError("down")

    client = RetryingClient(MockLLM(broken), max_retries=2, sleep=lambda s: None)
    with pytest.raises(ConnectionError):
        client.complete("s", "u")


def test_usage_cost():
    log = UsageLog(price_per_1k_prompt=1.0, price_per_1k_completion=2.0)
    llm = MockLLM(lambda s, u: "x" * 4000)
    log.add(llm.complete("", "y" * 2000))
    assert log.cost == pytest.approx(0.5 + 2.0)


def test_factory(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert isinstance(get_client(), MockLLM)
    with pytest.raises(ValueError):
        get_client("unknown")
