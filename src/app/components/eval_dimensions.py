import streamlit as st
from typing import Dict, Any
from src.cost_modelling.calculator import (
    calculate_storage_cost,
    calculate_checkpoint_storage_tb,
    calculate_project_compute_cost,
)


def render_eval_dimensions(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Render the eval dimensions form.
    Args:
        training_config: The training configuration dictionary.
    Returns:
        The training configuration dictionary with the eval dimensions added.
    """

    with st.expander("Experiment & Evaluation", expanded=True):
        exp_col1, exp_col2, exp_col3 = st.columns(3)

        with exp_col1:
            num_training_runs = st.number_input(
                "Full Training Runs",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                help="Number of complete training runs from scratch.",
            )
            num_checkpoints = st.number_input(
                "Checkpoints per Run",
                min_value=1,
                max_value=200,
                value=5,
                step=1,
                help="Number of model checkpoints saved to S3 per training run.",
            )

        with exp_col2:
            num_hp_trials = st.number_input(
                "HP Tuning Trials",
                min_value=0,
                max_value=1000,
                value=0,
                step=1,
                help="Number of hyperparameter search trials.",
            )
            hp_fraction = st.number_input(
                "HP Trial Length",
                min_value=0.05,
                max_value=1.0,
                value=0.30,
                step=0.05,
                format="%.2f",
                help="Typical HP trial is ~30% of a full run's cost.",
            )

        with exp_col3:
            num_ablations = st.number_input(
                "Ablation Studies",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
                help="Number of ablation experiments.",
            )
            ablation_fraction = st.number_input(
                "Ablation Length",
                min_value=0.05,
                max_value=1.0,
                value=0.50,
                step=0.05,
                format="%.2f",
                help="Typical ablation is ~50% of a full run's cost.",
            )

        storage_class = st.selectbox(
            "S3 Storage Class",
            options=["standard", "intelligent_tiering", "standard_ia", "glacier"],
            index=0,
            help="S3 storage class for dataset and checkpoints.",
        )
        training_config["S3 Storage Class"] = storage_class

        # Checkpoint size depends on training method
        ft_method = training_config.get("Fine-Tuning Method")
        if ft_method in ("LoRA", "QLoRA"):
            checkpoint_params = training_config.get("Trainable Parameters", 0)
        else:
            checkpoint_params = training_config.get(
                "Parameter Count",
                training_config.get("Base Model Params", 0),
            )
        checkpoint_size_tb = calculate_checkpoint_storage_tb(
            checkpoint_params=checkpoint_params,
            num_checkpoints=num_checkpoints,
            num_training_runs=num_training_runs,
            num_hp_trials=num_hp_trials,
            num_ablations=num_ablations,
        )

        single_run_cost = training_config["Compute Cost (USD)"]
        total_compute_cost = calculate_project_compute_cost(
            single_run_cost=single_run_cost,
            num_training_runs=num_training_runs,
            num_hp_trials=num_hp_trials,
            hp_fraction=hp_fraction,
            num_ablations=num_ablations,
            ablation_fraction=ablation_fraction,
        )
        dataset_storage_cost = calculate_storage_cost(
            training_config.get("Dataset Size (TB)", 0),
            training_config.get("Storage Duration (months)", 1),
            storage_class,
        )
        checkpoint_storage_cost = calculate_storage_cost(
            checkpoint_size_tb,
            training_config.get("Storage Duration (months)", 1),
            storage_class,
        )
        total_runs = num_training_runs + num_hp_trials + num_ablations

        training_config["Full Training Runs"] = num_training_runs
        training_config["HP Tuning Trials"] = num_hp_trials
        training_config["HP Run Fraction"] = hp_fraction
        training_config["Ablation Studies"] = num_ablations
        training_config["Ablation Run Fraction"] = ablation_fraction
        training_config["Total Experiment Runs"] = total_runs
        training_config["Num Checkpoints"] = num_checkpoints
        training_config["Checkpoint Storage (TB)"] = round(checkpoint_size_tb, 6)
        training_config["S3 Storage Class"] = storage_class
        training_config["Total Compute Cost (USD)"] = round(total_compute_cost, 2)
        training_config["Dataset Storage Cost (USD)"] = round(dataset_storage_cost, 2)
        training_config["Checkpoint Storage Cost (USD)"] = round(
            checkpoint_storage_cost, 2
        )
        training_config["Total Project Cost (USD)"] = round(
            total_compute_cost + dataset_storage_cost + checkpoint_storage_cost, 2
        )
    return training_config
