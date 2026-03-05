"""URL query-parameter persistence for Floply.

On first page load, URL params → session state (restores widget values).
After every render, session state → URL params (keeps the URL shareable).

Usage in app.py::

    from src.app.query_params import load_from_query_params, sync_to_query_params

    def main():
        load_from_query_params()   # must be first
        ...render tabs...
        sync_to_query_params()     # must be last
"""

import streamlit as st

# Keys to track, mapped to their Python types.
# `list` means the param can appear multiple times (multiselect widgets).
_TRACKED_KEYS: dict[str, type] = {
    # ── Active tab ───────────────────────────────────────────────────────────
    "active_tab": str,
    # ── Budget Optimizer ────────────────────────────────────────────────────
    "ms_budget_val": float,
    "ms_budget_unit": str,
    "ms_modality": str,
    "ms_training_type": str,
    "ms_ft_method": str,
    "ms_ft_base_model": str,
    "ms_lora_rank": int,
    "ms_lora_modules": list,
    "ms_explore_dir": str,
    "ms_dataset_slider": float,
    "ms_model_slider": float,
    "ms_rank_slider": int,
    "ms_epochs": int,
    "ms_hp_trials": int,
    "ms_hp_fraction_pct": int,
    "ms_storage_months": int,
    "ms_storage_class": str,
    "ms_num_instances": int,
    # ── Minimum Dataset Size ─────────────────────────────────────────────────
    "mds_training_type": str,
    "mds_pre_params_val": float,
    "mds_pre_params_unit": str,
    "mds_base_model": str,
    "mds_ft_params_val": float,
    "mds_ft_params_unit": str,
    "mds_ft_d_model": int,
    "mds_ft_num_layers": int,
    "mds_ft_method": str,
    "mds_lora_rank": int,
    "mds_target_modules": list,
    "mds_rc_modality": str,
    "mds_rc_avg_words": int,
    "mds_rc_resolution": int,
    "mds_rc_patch_size": int,
    "mds_rc_clip_duration": float,
    "mds_rc_audio_tokenizer": str,
    "mds_rc_codebooks": int,
    "mds_rc_vid_duration": float,
    "mds_rc_fps": int,
    "mds_rc_vid_resolution": int,
    "mds_rc_vid_patch_size": int,
    # ── Create A Training Budget — dataset ───────────────────────────────────
    "ce_modality": str,
    "ce_dataset_size_val": float,
    "ce_dataset_size_unit": str,
    "ce_avg_words": int,
    "ce_resolution": int,
    "ce_patch_size": int,
    "ce_clip_duration": float,
    "ce_audio_tok": str,
    "ce_codebooks": int,
    "ce_vid_duration": float,
    "ce_fps": int,
    "ce_vid_resolution": int,
    "ce_vid_patch_size": int,
    "ce_storage_months": int,
    # ── Create A Training Budget — training ──────────────────────────────────
    "ce_training_type": str,
    "ce_model_type": str,
    "ce_pre_parameter_count_val": float,
    "ce_pre_parameter_count_unit": str,
    "ce_pre_d_model": int,
    "ce_pre_num_layers": int,
    "ce_ft_type": str,
    "ce_base_model": str,
    "ce_ft_parameter_count_val": float,
    "ce_ft_parameter_count_unit": str,
    "ce_ft_d_model": int,
    "ce_ft_num_layers": int,
    "ce_ft_method": str,
    "ce_lora_rank": int,
    "ce_target_modules": list,
    "ce_rl_algo": str,
    # ── Create A Training Budget — compute ───────────────────────────────────
    "ce_epochs": int,
    "ce_grad_ckpt": bool,
    "ce_instance": str,
    "ce_num_instances": int,
    "ce_precision": str,
    "ce_mfu": int,
    "ce_batch_size": int,
    "ce_grad_acc": int,
    # ── Create A Training Budget — eval/experiment ───────────────────────────
    "ce_training_runs": int,
    "ce_checkpoints": int,
    "ce_hp_trials": int,
    "ce_hp_fraction": float,
    "ce_ablations": int,
    "ce_ablation_fraction": float,
    "ce_storage_class": str,
}


def load_from_query_params() -> None:
    """Populate session state from URL query params (once per browser session).

    Should be called at the very top of ``main()`` before any widgets render.
    Subsequent reruns skip the load so that user interactions are not overwritten.

    Sets ``st.session_state["_qp_has_state"] = True`` when valid state is
    loaded.  Pages that contain auto-cascade / reset logic (e.g. the Budget
    Optimizer) check this flag so they don't overwrite the restored values with
    fresh defaults on the first render.
    """
    if st.session_state.get("_qp_loaded"):
        return
    st.session_state["_qp_loaded"] = True

    params = st.query_params
    loaded_any = False
    for key, typ in _TRACKED_KEYS.items():
        if key not in params:
            continue
        try:
            if typ is list:
                values = params.get_all(key)
                if values:
                    st.session_state[key] = list(values)
                    loaded_any = True
            elif typ is bool:
                raw = params[key].lower()
                st.session_state[key] = raw in ("true", "1", "yes")
                loaded_any = True
            else:
                st.session_state[key] = typ(params[key])
                loaded_any = True
        except (ValueError, TypeError):
            pass

    # Signal to cascade logic that valid URL state was restored, so pages
    # should not overwrite it with computed defaults on the first render.
    if loaded_any:
        st.session_state["_qp_has_state"] = True


def clear_state() -> None:
    """Wipe all session state and the URL params.

    Call this then immediately call ``st.rerun()``.  The subsequent rerun will
    find an empty session state and no query params, so every widget reverts
    to its coded default.
    """
    st.query_params.clear()
    st.session_state.clear()


def sync_to_query_params() -> None:
    """Write tracked session-state values to URL query params.

    Should be called at the very end of ``main()`` after all widgets have
    rendered.  Writing to ``st.query_params`` does not trigger a rerun.
    """
    new_params: dict[str, str | list[str]] = {}

    for key, typ in _TRACKED_KEYS.items():
        val = st.session_state.get(key)
        if val is None:
            continue
        if typ is list:
            if val:
                new_params[key] = [str(v) for v in val]
        elif typ is bool:
            new_params[key] = "true" if val else "false"
        else:
            new_params[key] = str(val)

    # Remove stale keys then upsert new ones to keep the URL clean.
    for k in list(st.query_params.keys()):
        if k not in new_params:
            del st.query_params[k]
    for k, v in new_params.items():
        st.query_params[k] = v
