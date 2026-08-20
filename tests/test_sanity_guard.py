"""The guard between a filter regression and a confidently wrong cost estimate.

Absolute bounds catch values that cannot be right at all. Relative drift catches
the more dangerous case: a plausible number for the wrong product. A Windows or
capacity-block SKU is a multiple of the on-demand rate, not an absurd figure, so
only comparison against the previous value catches it.
"""

import pytest

from src.cost_modelling.pricing.models import InstanceHardware, RegionPricing
from src.cost_modelling.pricing.sanity import (
    check_drift,
    check_hardware,
    check_hourly_cost,
    check_storage_cost,
    is_acceptable_snapshot,
    validate_region_pricing,
)
from src.cost_modelling.pricing.settings import build_settings
from tests.test_cache import a_region


@pytest.fixture
def settings():
    return build_settings()


# -- Absolute bounds --------------------------------------------------------

@pytest.mark.parametrize("price", [55.04, 0.05, 1000.0, 26.87])
def test_plausible_hourly_prices_pass(price):
    assert check_hourly_cost("p5.48xlarge", "us-east-1", price) is None


@pytest.mark.parametrize("price", [0.0, -1.0, 0.001, 9999.0, float("inf"), float("nan")])
def test_implausible_hourly_prices_are_rejected(price):
    assert check_hourly_cost("p5.48xlarge", "us-east-1", price) is not None


def test_rejection_message_names_the_instance_and_region():
    reason = check_hourly_cost("p5.48xlarge", "eu-west-1", 9999.0)
    assert "p5.48xlarge" in reason and "eu-west-1" in reason


@pytest.mark.parametrize("price", [23.0, 4.0, 0.1, 500.0])
def test_plausible_storage_prices_pass(price):
    assert check_storage_cost("standard", "us-east-1", price) is None


@pytest.mark.parametrize("price", [0.0, -5.0, 0.01, 5000.0])
def test_implausible_storage_prices_are_rejected(price):
    assert check_storage_cost("standard", "us-east-1", price) is not None


# -- Relative drift ---------------------------------------------------------

def test_no_previous_value_means_no_drift_check():
    assert check_drift("p5/us-east-1", 55.04, None, max_drift=4.0) is None


def test_the_real_p5_price_cut_passes():
    """AWS cut p5 to ~0.56x of list; a guard that rejects that is useless."""
    assert check_drift("p5/us-east-1", 55.04, 98.32, max_drift=4.0) is None


@pytest.mark.parametrize("ratio", [1.0, 1.5, 3.9, 0.3])
def test_ordinary_movements_pass(ratio):
    assert check_drift("p5/us-east-1", 50.0 * ratio, 50.0, max_drift=4.0) is None


def test_a_ten_times_jump_is_rejected():
    reason = check_drift("p5/us-east-1", 500.0, 50.0, max_drift=4.0)
    assert reason is not None
    assert "10.0x" in reason and "filters" in reason


def test_a_ten_times_drop_is_rejected():
    assert check_drift("p5/us-east-1", 5.0, 50.0, max_drift=4.0) is not None


def test_zero_previous_value_is_not_divided_by():
    assert check_drift("p5/us-east-1", 55.0, 0.0, max_drift=4.0) is None


# -- Hardware consistency ---------------------------------------------------

def test_consistent_hardware_passes():
    assert check_hardware(
        InstanceHardware("p5.48xlarge", "H100", 8, 80, 640, 192, 2048, 3200)) is None


def test_total_gpu_memory_must_match_count_times_per_gpu():
    defect = check_hardware(
        InstanceHardware("p5.48xlarge", "H100", 8, 80, 999, 192, 2048, 3200))
    assert defect is not None and "total_gpu_memory" in defect


def test_rounding_slack_is_tolerated():
    assert check_hardware(
        InstanceHardware("p5.48xlarge", "H100", 8, 80, 636, 192, 2048, 3200)) is None


def test_missing_gpu_name_is_rejected():
    assert check_hardware(
        InstanceHardware("p5.48xlarge", "", 8, 80, 640, 192, 2048, 3200)) is not None


@pytest.mark.parametrize("count", [0, 128])
def test_implausible_gpu_counts_are_rejected(count):
    assert check_hardware(
        InstanceHardware("x", "H100", count, 80, 80 * max(count, 1), 8, 64, 100)) is not None


# -- Whole-region validation ------------------------------------------------

def test_bad_price_falls_back_to_the_cached_value(settings):
    previous = a_region(prices={"p5.48xlarge": 55.04, "p4d.24xlarge": 26.87,
                                "p3.16xlarge": 24.48})
    fetched = a_region(prices={"p5.48xlarge": 9999.0, "p4d.24xlarge": 26.87,
                               "p3.16xlarge": 24.48})

    result = validate_region_pricing(fetched, previous, settings)

    assert result.prices["p5.48xlarge"] == pytest.approx(55.04), "kept the trusted value"
    assert result.prices["p4d.24xlarge"] == pytest.approx(26.87)
    assert any("9,999" in w or "9999" in w for w in result.warnings)


def test_bad_price_with_no_cached_value_is_dropped_entirely(settings):
    fetched = a_region(prices={"p5.48xlarge": 9999.0, "p4d.24xlarge": 26.87,
                               "p3.16xlarge": 24.48})
    result = validate_region_pricing(fetched, None, settings)

    assert "p5.48xlarge" not in result.prices, "never surface a value we rejected"
    assert result.warnings


def test_rejections_are_never_silent(settings):
    fetched = a_region(prices={"p5.48xlarge": 0.0})
    result = validate_region_pricing(fetched, None, settings)
    assert len(result.warnings) >= 1


def test_offered_list_tracks_surviving_prices(settings):
    fetched = a_region(prices={"p5.48xlarge": 9999.0, "p4d.24xlarge": 26.87})
    result = validate_region_pricing(fetched, None, settings)
    assert result.offered == ("p4d.24xlarge",)


def test_storage_drift_is_guarded_too(settings):
    previous = a_region(storage={"standard": 23.0})
    fetched = a_region(storage={"standard": 230.0})

    result = validate_region_pricing(fetched, previous, settings)
    assert result.storage["standard"] == pytest.approx(23.0)
    assert any("standard" in w for w in result.warnings)


# -- Acceptance gate --------------------------------------------------------

def test_full_region_is_acceptable(settings):
    ok, reason = is_acceptable_snapshot(a_region(), settings)
    assert ok and reason is None


def test_thin_region_is_not_acceptable(settings):
    ok, reason = is_acceptable_snapshot(a_region(prices={"p5.48xlarge": 55.04}), settings)
    assert not ok
    assert str(settings.min_instances) in reason


def test_empty_region_is_not_acceptable(settings):
    ok, _ = is_acceptable_snapshot(a_region(prices={}), settings)
    assert not ok
