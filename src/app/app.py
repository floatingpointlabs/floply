import os
import streamlit as st
from streamlit.components.v1 import html

from src.app.config import configure_page, PAGE_TITLE, _TAB_CSS
from src.app.page_renderers.about_page import render_about_page
from src.app.page_renderers.cost_estimator_page import (
    render_create_a_training_budget_page,
)
from src.app.page_renderers.model_size_page import render_budget_optimizer_page
from src.app.page_renderers.min_dataset_size_page import render_min_dataset_size_page
from src.app.query_params import (
    load_from_query_params,
    sync_to_query_params,
    clear_state,
)


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
    load_from_query_params()

    configure_page()
    configure_analytics()

    st.markdown(_TAB_CSS, unsafe_allow_html=True)
    st.markdown(
        f"<h1 style='text-align: center;'>{PAGE_TITLE}</h1>", unsafe_allow_html=True
    )
    if st.button("Clear", help="Reset all settings and clear the shared URL"):
        clear_state()
        # This is genuinely stupid. Streamlit's html() runs in a sandboxed iframe
        # that blocks programmatic top-level navigation, so window.parent.location
        # .replace() doesn't work directly. Instead we have to escape the sandbox
        # by injecting a <script> tag into the parent document's <head> via
        # same-origin DOM access, so it runs outside the sandbox. All of this just
        # to reload the page without query params.
        html(
            "<script>"
            "var s=window.parent.document.createElement('script');"
            "s.textContent='window.location.replace(window.location.pathname)';"
            "window.parent.document.head.appendChild(s);"
            "</script>",
            height=0,
        )

    _TAB_LABELS = [
        "Budget Optimizer",
        "Minimum Data Calculator",
        "Create A Training Budget",
        "About",
    ]
    _stored_tab = st.session_state.get("active_tab")
    _default_tab = _stored_tab if _stored_tab in _TAB_LABELS else None

    tab1, tab2, tab3, tab4 = st.tabs(  # type: ignore[call-overload]
        _TAB_LABELS,
        key="active_tab",  # type: ignore[call-overload]
        on_change="rerun",  # type: ignore[call-overload]
        default=_default_tab,
    )

    with tab1:
        render_budget_optimizer_page()
    with tab2:
        render_min_dataset_size_page()
    with tab3:
        render_create_a_training_budget_page()
    with tab4:
        render_about_page()

    sync_to_query_params()


if __name__ == "__main__":
    main()
