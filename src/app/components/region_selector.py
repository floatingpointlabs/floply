"""Sidebar region selector and pricing provenance.

Region is a whole-app choice: it changes instance prices, S3 storage costs, and
which instance types are offered. It lives in the sidebar — previously an empty
panel — so one selection applies across all four tabs.
"""

import streamlit as st

from src.app.config import AWS_REGION_LABELS
from src.cost_modelling.gpu_specs import (
    get_provenance,
    invalidate_catalog,
    list_regions,
)
from src.cost_modelling.pricing.settings import get_settings

SESSION_KEY = "floply_region"


def current_region() -> str:
    """The selected region, or the configured default before the sidebar renders."""
    return st.session_state.get(SESSION_KEY) or get_settings().default_region


def _format_region(code: str) -> str:
    label = AWS_REGION_LABELS.get(code)
    return f"{code} — {label}" if label else code


def render_region_sidebar() -> str:
    """Render the region selector and provenance chip. Returns the region code."""
    regions = list_regions()

    with st.sidebar:
        st.caption("AWS PRICING")

        current = current_region()
        region = st.selectbox(
            "Region",
            options=regions,
            index=regions.index(current) if current in regions else 0,
            format_func=_format_region,
            key=SESSION_KEY,
            help="Instance and S3 prices are region-specific.",
        )

        _render_provenance(region)

    return region


def _render_provenance(region: str) -> None:
    """Show where the numbers came from — never let a cost hide its source."""
    provenance = get_provenance(region)
    summary = provenance.summary()

    if not provenance.is_usable:
        st.error(summary, icon="⚠️")
    elif provenance.stale:
        st.warning(summary, icon="⏳")
    elif provenance.source == "live":
        st.success(summary, icon="✅")
    else:
        st.info(summary, icon="💾")

    st.caption("On-demand list price, Linux, shared tenancy.")

    with st.expander("Pricing data source"):
        st.markdown(
            f"**Source** `{provenance.source}`  \n"
            f"**Region** `{region}`  \n"
            f"**Fetched** "
            f"{provenance.fetched_at.strftime('%Y-%m-%d %H:%M UTC') if provenance.fetched_at else '—'}  \n"
            f"**Cache** `{provenance.store_description or '—'}`"
        )

        if provenance.last_error:
            st.markdown(f"**Last error** {provenance.last_error}")

        probe = st.session_state.get("floply_probe")
        if probe is not None and not probe.ok:
            st.markdown(f"**Startup check failed** {probe.detail}")
            if probe.missing_action:
                st.markdown(f"Add `{probe.missing_action}` to the IAM policy.")

        if provenance.quarantined:
            st.markdown("**Discovered but unusable**")
            for instance_type, reason in sorted(provenance.quarantined.items()):
                st.markdown(f"- `{instance_type}` — {reason}")

        if provenance.warnings:
            st.markdown("**Warnings**")
            for warning in provenance.warnings:
                st.markdown(f"- {warning}")

        if st.button("Refresh pricing now", key="refresh_pricing"):
            invalidate_catalog(region)
            st.rerun()


def render_unavailable(region: str, detail: str) -> None:
    """Full-width explanation when there is no pricing at all.

    Online-only means a cold cache plus an unreachable AWS is a real state, and
    showing a calculator that cannot price would be worse than saying so.
    """
    st.error(
        f"**Pricing unavailable for {region}.**\n\n{detail}\n\n"
        f"Floply reads live AWS pricing and has no bundled fallback. Check that "
        f"credentials are configured and that the IAM policy allows "
        f"`pricing:GetProducts`, `ec2:DescribeInstanceTypes` and "
        f"`ec2:DescribeInstanceTypeOfferings`.",
        icon="🚫",
    )
    st.caption(
        "Prime the cache from the command line with: "
        "`python -m src.cost_modelling.pricing.refresh_cli`"
    )
