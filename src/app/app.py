import streamlit as st

from src.app.config import configure_page
from src.app.page_renderers.about_page import render_about_page
from src.app.page_renderers.cost_estimator_page import render_model_training_cost_estimator_page


def main():
    # Setup
    configure_page()

    # Title (Centered)
    st.markdown("<h1 style='text-align: center;'>Floply</h1>", unsafe_allow_html=True)

    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Model Training Cost Estimator", "About"],
        index=0,
        label_visibility="visible",
    )

    # Page Selector
    match page:
        case "Model Training Cost Estimator":
            render_model_training_cost_estimator_page()
        case "About":   
            render_about_page()

if __name__ == "__main__":
    main()
