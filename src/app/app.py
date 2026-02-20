"""ML Training Cost Estimator - Streamlit Application

This application provides cost estimation for training machine learning models
on AWS GPU instances.
"""

import streamlit as st
from src.cost_modelling.calculator import (
    calculate_single_run_cost,
    calculate_project_cost,
)
from src.cost_modelling.scenarios import (
    ALL_SCENARIOS,
    compare_scenarios,
)
from src.app.config import configure_page, apply_custom_css
from src.app.components import (
    render_model_config_form,
    render_training_config_form,
    render_project_config_form,
    render_cost_summary,
    render_cost_breakdown_table,
    render_efficiency_metrics,
    render_project_cost_summary,
    render_comparison_table,
    render_scenario_cards,
    render_export_buttons,
    create_cost_breakdown_pie_chart,
    create_project_cost_breakdown_chart,
    create_scenario_comparison_chart,
    create_duration_comparison_chart,
    create_cost_vs_model_size_chart,
    create_gpu_utilization_timeline,
    create_cost_efficiency_comparison,
)


def main():
    """Main application entry point."""
    configure_page()
    apply_custom_css()
    
    # Title and description
    st.title("💰 Floply — ML Training Cost Estimator")
    st.markdown("""
    Estimate the cost of training machine learning models on AWS GPU instances.
    Based on industry-standard FLOPs calculations and real AWS pricing data.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Single Run Estimator", "Scenario Comparison", "Project Budget Planner", "About"]
    )
    
    if page == "Single Run Estimator":
        render_single_run_page()
    elif page == "Scenario Comparison":
        render_scenario_comparison_page()
    elif page == "Project Budget Planner":
        render_project_planner_page()
    elif page == "About":
        render_about_page()


def render_single_run_page():
    """Render the single training run estimator page."""
    st.header("Single Training Run Estimator")
    st.markdown("Configure your model and training setup to estimate costs.")
    
    # Input forms
    with st.expander("📊 Model Configuration", expanded=True):
        model_config = render_model_config_form()
    
    with st.expander("⚙️ Training Configuration", expanded=True):
        training_config = render_training_config_form(model_config)
    
    # Calculate button
    if st.button("Calculate Cost", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
            cost_breakdown = calculate_single_run_cost(model_config, training_config)
        
        # Display results
        st.success("✅ Calculation complete!")
        
        # Summary metrics
        render_cost_summary(cost_breakdown)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_cost_breakdown_pie_chart(cost_breakdown)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_gpu_utilization_timeline(cost_breakdown)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown
        render_cost_breakdown_table(cost_breakdown)
        
        # Efficiency metrics
        render_efficiency_metrics(cost_breakdown)
        
        # Save to session state for project planner
        st.session_state['last_cost_breakdown'] = cost_breakdown


def render_scenario_comparison_page():
    """Render the scenario comparison page."""
    st.header("Predefined Scenario Comparison")
    st.markdown("""
    Compare pre-configured training scenarios representing common workloads:
    - **Quick Validation**: Test training loop (100M params, 1B tokens)
    - **Small-Scale Research**: Proof of concept (350M params, 50B tokens)
    - **Emerging Capabilities**: See emergent behavior (1B params, 200B tokens)
    - **Production Training**: Full-scale deployment (7B params, 1T tokens)
    - **Medium Research**: Advanced research (3B params, 500B tokens)
    - **Large-Scale Training**: Large model (13B params, 1.3T tokens)
    """)
    
    # Scenario selection
    scenario_options = st.multiselect(
        "Select scenarios to compare:",
        options=list(ALL_SCENARIOS.keys()),
        default=["quick_validation", "small_scale_research", "emerging_capabilities", "production_training"],
        format_func=lambda x: ALL_SCENARIOS[x].name
    )
    
    if scenario_options:
        # Get scenario data
        scenarios = compare_scenarios(scenario_options)
        
        # Display scenario cards
        render_scenario_cards(scenarios)
        
        st.divider()
        
        # Comparison table
        df = render_comparison_table(scenarios)
        
        # Export buttons
        render_export_buttons(df, filename="scenario_comparison")
        
        st.divider()
        
        # Comparison charts
        st.subheader("Visual Comparisons")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_scenario_comparison_chart(scenarios)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_duration_comparison_chart(scenarios)
            st.plotly_chart(fig, use_container_width=True)
        
        if len(scenarios) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_cost_vs_model_size_chart(scenarios)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_cost_efficiency_comparison(scenarios)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one scenario to compare.")


def render_project_planner_page():
    """Render the project budget planner page."""
    st.header("Project Budget Planner")
    st.markdown("""
    Plan your complete project budget including multiple training runs,
    hyperparameter tuning, ablation studies, and data storage costs.
    """)
    
    # Check if we have a cost breakdown from single run estimator
    if 'last_cost_breakdown' in st.session_state:
        st.info("Using configuration from Single Run Estimator. You can modify project parameters below.")
        cost_breakdown = st.session_state['last_cost_breakdown']
    else:
        st.warning("⚠️ Please calculate a single training run cost first in the 'Single Run Estimator' page.")
        
        # Quick scenario selector
        st.subheader("Or select a predefined scenario:")
        scenario_key = st.selectbox(
            "Choose a scenario:",
            options=list(ALL_SCENARIOS.keys()),
            format_func=lambda x: ALL_SCENARIOS[x].name
        )
        
        scenario = ALL_SCENARIOS[scenario_key]
        cost_breakdown = scenario.cost_breakdown
        
        st.info(f"Selected: {scenario.name} - {scenario.description}")
    
    # Project configuration
    with st.expander("📋 Project Configuration", expanded=True):
        project_config = render_project_config_form()
    
    # Calculate button
    if st.button("Calculate Project Cost", type="primary", use_container_width=True):
        with st.spinner("Calculating project costs..."):
            project_cost = calculate_project_cost(
                single_run_cost=cost_breakdown,
                **project_config
            )
        
        st.success("✅ Project cost calculated!")
        
        # Display project summary
        render_project_cost_summary(project_cost)
        
        # Project cost breakdown chart
        fig = create_project_cost_breakdown_chart(project_cost)
        st.plotly_chart(fig, use_container_width=True)
        
        # Budget recommendations
        st.subheader("💡 Budget Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Recommended Contingency (20%)",
                value=f"${project_cost['total_cost'] * 0.2:,.2f}",
                help="Additional budget buffer for unexpected costs"
            )
        
        with col2:
            total_with_contingency = project_cost['total_cost'] * 1.2
            st.metric(
                label="Total with Contingency",
                value=f"${total_with_contingency:,.2f}",
                help="Total budget including 20% contingency"
            )


def render_about_page():
    """Render the about page with methodology and assumptions."""
    st.header("About This Tool")
    
    st.markdown("""
    ## Overview
    
    This tool estimates the cost of training machine learning models on AWS GPU instances
    using industry-standard FLOPs calculations and real AWS pricing data (as of February 2026).
    
    ## Methodology
    
    ### FLOPs Calculation
    
    We use the standard formula for training compute:
    
    **C = 6 × N × D**
    
    Where:
    - **C** = Total FLOPs required
    - **N** = Number of model parameters
    - **D** = Number of training tokens
    - **6** = Multiplier accounting for:
      - Forward pass: 2N FLOPs
      - Backward pass: 4N FLOPs
      - Optimizer overhead
    
    ### GPU Time Estimation
    
    **GPU-hours = (Total FLOPs ÷ GPU throughput) ÷ Number of GPUs**
    
    We account for **Model FLOPs Utilization (MFU)** - the percentage of theoretical
    peak performance actually achieved in practice:
    - A100 GPUs: ~45% MFU (typical for transformers)
    - H100 GPUs: ~55% MFU (improved efficiency)
    
    ### Cost Calculation
    
    **Total Cost = GPU-hours × Hourly Instance Rate**
    
    ## AWS Pricing (February 2026)
    
    | Instance Type | GPU | GPU Count | Memory/GPU | Cost/Hour |
    |---------------|-----|-----------|------------|-----------|
    | p4d.24xlarge | A100 | 8 | 40GB | $26.87 |
    | p4de.24xlarge | A100 | 8 | 80GB | $32.77 |
    | p5.48xlarge | H100 | 8 | 80GB | $66.64 |
    | p3.16xlarge | V100 | 8 | 16GB | $24.48 |
    | p3dn.24xlarge | V100 | 8 | 32GB | $31.22 |
    
    *Prices are averages across US regions for on-demand instances*
    
    ## Assumptions
    
    - **Architecture**: Transformer-based models (6× FLOPs multiplier)
    - **Precision**: Mixed precision training (FP16/BF16)
    - **Pricing**: On-demand instances (no Spot or Reserved pricing)
    - **Region**: US-East-1 as baseline
    - **Scaling**: Linear scaling (no communication overhead modeling)
    - **Optimizer**: Adam optimizer (standard for most LLM training)
    
    ## Accuracy
    
    This tool aims for **10-20% accuracy** compared to actual training costs.
    Variations can occur due to:
    - Code optimization quality
    - Framework efficiency (PyTorch, JAX, etc.)
    - Network communication overhead
    - Data loading bottlenecks
    - Actual vs. advertised GPU performance
    
    ## Data Sources
    
    - AWS pricing: [aws.amazon.com/ec2/pricing](https://aws.amazon.com/ec2/pricing)
    - FLOPs calculations: Academic papers (Chinchilla, GPT-3, LLaMA)
    - MFU benchmarks: Industry training reports
    
    ## Validation
    
    Cost estimates have been validated against published training costs for:
    - GPT-3 (175B parameters)
    - LLaMA models (7B-70B parameters)
    - Chinchilla (70B parameters)
    
    ## Limitations
    
    - Does not account for data preprocessing costs
    - Assumes steady-state training (no startup overhead)
    - Does not model pipeline parallelism efficiency
    - Storage costs are simplified (S3 only)
    - No multi-cloud comparison (AWS only)
    
    ## Future Enhancements
    
    Planned features:
    - Multi-cloud support (GCP, Azure)
    - Spot instance pricing
    - Data acquisition cost modeling
    - Carbon footprint calculator
    - Historical cost tracking
    - API endpoint for programmatic access
    
    ## Contact & Feedback
    
    For questions, suggestions, or bug reports, please contact the development team.
    
    ---
    
    **Version**: 1.0.0  
    **Last Updated**: February 2026
    """)
    
    # Quick reference card
    with st.expander("📚 Quick Reference: Common Model Sizes"):
        st.markdown("""
        | Model Type | Parameters | Typical Use Case |
        |------------|------------|------------------|
        | Small | 100M-350M | Research, prototyping, debugging |
        | Medium | 1B-3B | Advanced research, specialized tasks |
        | Large | 7B-13B | General-purpose, production deployments |
        | Very Large | 30B-70B+ | Frontier models, highest capability |
        
        **Training Data Guidelines (Chinchilla Optimal):**
        - 100M params → 2B tokens
        - 1B params → 20B tokens
        - 7B params → 140B tokens
        - 13B params → 260B tokens
        """)


if __name__ == "__main__":
    main()
