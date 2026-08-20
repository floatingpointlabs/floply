import logging
import os

import streamlit as st
from streamlit.components.v1 import html

from src.cost_modelling.pricing.refresh import probe
from src.cost_modelling.pricing.settings import get_settings

from src.app.components.region_selector import render_region_sidebar, render_unavailable
from src.app.config import configure_page, PAGE_TITLE, _TAB_CSS
from src.cost_modelling.gpu_specs import PricingUnavailableError, get_provenance
from src.app.page_renderers.about_page import render_about_page
from src.app.page_renderers.cost_estimator_page import render_create_a_training_budget_page
from src.app.page_renderers.model_size_page import render_budget_optimizer_page
from src.app.page_renderers.min_dataset_size_page import render_min_dataset_size_page


@st.cache_resource
def run_startup_probe():
    """One cheap credential check at startup, cached for the process lifetime.

    Fetching is otherwise fully lazy, so without this a broken IAM policy is
    discovered by a user several form sections deep rather than at deploy time.
    """
    settings = get_settings()
    if not settings.startup_probe:
        return None

    result = probe(settings)
    st.session_state["floply_probe"] = result
    if result.ok:
        logging.getLogger(__name__).info("AWS pricing probe: %s", result.detail)
    else:
        logging.getLogger(__name__).warning(
            "AWS pricing probe failed: %s%s",
            result.detail,
            f" (missing {result.missing_action})" if result.missing_action else "",
        )
    return result


@st.cache_resource
def configure_analytics():
    umami_url = os.environ.get("UMAMI_URL")
    website_id = os.environ.get("UMAMI_WEBSITE_ID")

    if umami_url and website_id:
        html(
            f'<script async defer src="{umami_url}" data-website-id="{website_id}"></script>',
            height=0,
            width=0,
        )


def main():
    configure_page()
    configure_analytics()
    run_startup_probe()

    st.markdown(_TAB_CSS, unsafe_allow_html=True)
    st.markdown(
        f"<h1 style='text-align: center;'>{PAGE_TITLE}</h1>", unsafe_allow_html=True
    )

    region = render_region_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Budget Optimizer",
            "Minimum Data Calculator",
            "Create A Training Budget",
            "About",
        ]
    )

    # The About tab needs no pricing, so it stays available even when AWS is
    # unreachable. The three calculators cannot honestly render without it, and
    # say so up front rather than letting someone fill in a form first.
    provenance = get_provenance(region)

    def _guarded(render):
        if not provenance.is_usable:
            render_unavailable(region, provenance.last_error or "No cached pricing.")
            return
        try:
            render()
        except PricingUnavailableError as exc:
            render_unavailable(exc.region, exc.detail)

    with tab1:
        _guarded(render_budget_optimizer_page)
    with tab2:
        _guarded(render_min_dataset_size_page)
    with tab3:
        _guarded(render_create_a_training_budget_page)
    with tab4:
        render_about_page()


if __name__ == "__main__":
    main()
