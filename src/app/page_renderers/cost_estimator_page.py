from src.app.page_renderers._shared import _render_standard_training_page
from src.app.components.output_dimensions import render_estimate_summary_numbers


def render_estimate_page() -> None:
    _render_standard_training_page(
        output_fn=render_estimate_summary_numbers,
        subtitle="Given your model, dataset, and hardware — how much will training cost and how long will it take?",
    )
