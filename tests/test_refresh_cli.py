"""The refresh CLI: exit codes and human-readable output.

This is the tool someone reaches for when prices look wrong, so its output has
to show what was fetched, what was skipped, and why — without a traceback.
"""

from datetime import datetime, timezone

import pytest

from src.cost_modelling.pricing import refresh as refresh_module
from src.cost_modelling.pricing.models import ProbeResult, RegionPricing, SpecsSnapshot
from src.cost_modelling.pricing.refresh import RefreshResult
from src.cost_modelling.pricing.refresh_cli import main
from tests.test_refresh import hardware


@pytest.fixture
def ok_probe(monkeypatch):
    monkeypatch.setattr(
        refresh_module, "probe", lambda *_a, **_k: ProbeResult(True, "reachable"))


def _result(**overrides) -> RefreshResult:
    now = datetime.now(timezone.utc)
    defaults = dict(
        ok=True,
        specs=SpecsSnapshot(fetched_at=now, instances={"p5.48xlarge": hardware()}),
        regions={"us-east-1": RegionPricing(
            region="us-east-1", fetched_at=now,
            prices={"p5.48xlarge": 55.04}, storage={"standard": 23.0})},
        store_description="/app/.cache/floply",
    )
    defaults.update(overrides)
    return RefreshResult(**defaults)


def test_probe_failure_exits_two_without_fetching(monkeypatch, capsys):
    monkeypatch.setattr(
        refresh_module, "probe",
        lambda *_a, **_k: ProbeResult(False, "No AWS credentials found."))

    def explode(*_a, **_k):
        raise AssertionError("must not fetch after a failed probe")
    monkeypatch.setattr(refresh_module, "refresh_all", explode)

    assert main([]) == 2
    assert "No AWS credentials found." in capsys.readouterr().out


def test_probe_failure_names_the_missing_iam_action(monkeypatch, capsys):
    monkeypatch.setattr(
        refresh_module, "probe",
        lambda *_a, **_k: ProbeResult(False, "denied", missing_action="pricing:GetProducts"))

    assert main([]) == 2
    assert "pricing:GetProducts" in capsys.readouterr().out


def test_probe_only_skips_the_fetch(monkeypatch, ok_probe, capsys):
    def explode(*_a, **_k):
        raise AssertionError("must not fetch in --probe-only mode")
    monkeypatch.setattr(refresh_module, "refresh_all", explode)

    assert main(["--probe-only"]) == 0
    assert "ok" in capsys.readouterr().out


def test_successful_refresh_prints_prices(monkeypatch, ok_probe, capsys):
    monkeypatch.setattr(refresh_module, "refresh_all", lambda **_k: _result())

    assert main(["--regions", "us-east-1"]) == 0
    out = capsys.readouterr().out
    assert "p5.48xlarge" in out
    assert "55.04" in out
    assert "/app/.cache/floply" in out


def test_quarantined_instances_are_listed_with_the_fix(monkeypatch, ok_probe, capsys):
    monkeypatch.setattr(
        refresh_module, "refresh_all",
        lambda **_k: _result(quarantined={"p9.future": 'no curated FLOPs for GPU "Z900".'}))

    main(["--regions", "us-east-1"])
    out = capsys.readouterr().out
    assert "p9.future" in out
    assert "gpu_hardware.yaml" in out


def test_warnings_are_surfaced(monkeypatch, ok_probe, capsys):
    monkeypatch.setattr(
        refresh_module, "refresh_all",
        lambda **_k: _result(warnings=["us-east-1: 3 SKUs matched, took the minimum"]))

    main(["--regions", "us-east-1"])
    assert "3 SKUs matched" in capsys.readouterr().out


def test_partial_failure_exits_one(monkeypatch, ok_probe, capsys):
    monkeypatch.setattr(
        refresh_module, "refresh_all",
        lambda **_k: _result(ok=False, warnings=["eu-west-1: prices could not be refreshed"]))

    assert main(["--regions", "us-east-1,eu-west-1"]) == 1
    assert "not refreshed" in capsys.readouterr().out


def test_regions_flag_overrides_the_configured_list(monkeypatch, ok_probe):
    seen = {}

    def record(**kwargs):
        seen.update(kwargs)
        return _result()
    monkeypatch.setattr(refresh_module, "refresh_all", record)

    main(["--regions", "us-east-1, eu-west-1"])
    assert seen["regions"] == ("us-east-1", "eu-west-1")


def test_force_flag_is_passed_through(monkeypatch, ok_probe):
    seen = {}

    def record(**kwargs):
        seen.update(kwargs)
        return _result()
    monkeypatch.setattr(refresh_module, "refresh_all", record)

    main(["--force"])
    assert seen["force"] is True
