"""End-to-end render smoke tests.

The app has no other UI coverage, and the precision selector is now driven by
per-GPU data rather than a fixed list — so a curation mistake could surface as
a page-level exception rather than a wrong number. These drive the real
Streamlit script and assert it renders clean.
"""

import pytest
from streamlit.testing.v1 import AppTest

APP = "src/app/app.py"


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=90).run()


def test_app_renders_without_exceptions(app):
    assert not app.exception, [str(e.value) for e in app.exception]


def test_budget_optimizer_auto_configures_hardware(app):
    """Exercises _auto_configure_hardware, which picks a precision per GPU."""
    from src.app.page_renderers.model_size_page import _auto_configure_hardware

    hw = _auto_configure_hardware.__wrapped__("us-east-1", "stamp")
    assert hw["instance_type"] == "p5.48xlarge"      # best TFLOPS/$ of the five
    assert hw["mixed_precision"] == "bf16"           # H100 supports it
    assert hw["peak_flops_per_gpu"] == pytest.approx(989e12)


def test_auto_config_falls_back_to_fp16_on_volta(monkeypatch):
    """A Volta-only catalog must not ask for bf16, which Volta lacks."""
    from src.app.page_renderers import model_size_page as page

    monkeypatch.setattr(page, "list_available_instances", lambda _region: ["p3.16xlarge"])
    hw = page._auto_configure_hardware.__wrapped__("us-east-1", "stamp")
    assert hw["instance_type"] == "p3.16xlarge"
    assert hw["mixed_precision"] == "fp16"
    assert hw["peak_flops_per_gpu"] == pytest.approx(125e12)


def test_auto_config_is_keyed_on_region_and_catalog_stamp():
    """Without both in the cache key it returned one region's answer forever."""
    import inspect

    from src.app.page_renderers.model_size_page import _auto_configure_hardware

    params = list(inspect.signature(_auto_configure_hardware.__wrapped__).parameters)
    assert params == ["region", "catalog_stamp"]


@pytest.fixture
def compute_form():
    """Drive the Create-A-Training-Budget flow far enough to reach the hardware widgets.

    AppTest flattens widgets across all four tabs, so 'Training Type' appears
    several times; the one belonging to this flow is the last, rendered only
    after a modality is chosen.
    """
    at = AppTest.from_file(APP, default_timeout=90).run()
    next(s for s in at.selectbox if s.label == "Select Modality").select("Text").run()
    [s for s in at.selectbox if s.label == "Training Type"][-1].select("Pre-Training").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_instances_are_listed_most_capable_first(compute_form):
    """Ordered by throughput, not price: a price cut must not move the default.

    AppTest reports `options` as the rendered labels and `value` as the
    underlying option, so ordering is asserted against the label prefixes.
    """
    instance = next(s for s in compute_form.selectbox if s.label == "Instance Type")
    assert instance.options[0].startswith("p5.48xlarge")     # H100
    assert instance.options[-1].startswith("p3dn.24xlarge")  # V100, legacy


def test_compute_section_offers_only_supported_precisions(compute_form):
    precision = next(s for s in compute_form.selectbox if s.label == "Mixed Precision")

    # The old global list advertised fp4 on every instance regardless of die.
    assert "fp4" not in precision.options
    assert precision.options == ["int8", "fp8", "bf16", "fp16", "tf32", "fp32"]
    assert precision.value == "bf16"


def test_instance_widget_stores_the_type_not_the_label(compute_form):
    """The label carries the price; storing it meant a refresh reset the choice."""
    instance = next(s for s in compute_form.selectbox if s.label == "Instance Type")

    assert instance.value == "p5.48xlarge", "widget state must be price-independent"
    assert "$" not in instance.value
    assert "$" in instance.options[0], "the rendered label still shows the price"


def test_precision_options_track_the_selected_instance(compute_form):
    """Switching to A100 drops fp8; switching to V100 drops bf16 and tf32 too.

    Widgets must be re-fetched after every run() — changing the option list
    changes the auto-generated widget id, so a held reference goes stale.
    """
    def precision():
        return next(s for s in compute_form.selectbox if s.label == "Mixed Precision")

    def instance():
        return next(s for s in compute_form.selectbox if s.label == "Instance Type")

    instance().select("p4d.24xlarge").run()
    assert "fp8" not in precision().options        # Ampere has no fp8
    assert "fp4" not in precision().options

    instance().select("p3.16xlarge").run()
    assert precision().options == ["int8", "fp16", "fp32"]
    assert precision().value == "fp16"             # Volta has no bf16 to default to
    assert not compute_form.exception


def test_full_flow_produces_a_total_cost(compute_form):
    """The whole pipeline still resolves end to end after the data split."""
    assert not compute_form.exception
    assert any("Total" in str(m.label) for m in compute_form.metric)
