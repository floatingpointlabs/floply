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

    st.markdown(_TAB_CSS, unsafe_allow_html=True)
    st.markdown(
        f"<h1 style='text-align: center;'>{PAGE_TITLE}</h1>", unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Budget Optimizer",
            "Minimum Data Calculator",
            "Create A Training Budget",
            "About",
        ]
    )

    with tab1:
        render_budget_optimizer_page()
    with tab2:
        render_min_dataset_size_page()
    with tab3:
        render_create_a_training_budget_page()
    with tab4:
        render_about_page()


if __name__ == "__main__":
    main()
