"""Budget Optimizer solve — pure functions, no Streamlit, no numpy.

Extracted verbatim from ``src/app/page_renderers/model_size_page.py`` so the solve can be
exercised without a UI attached. The page is now a widget shell that feeds
:class:`OptimizerInputs` in and renders :class:`BudgetOptimum` / :class:`SelectionResult` out.

Two-stage by design, mirroring how the page reads:

1. :func:`solve_budget_optimum` — where the budget lands you if you accept the scaling law.
2. :func:`resolve_selection`   — where the user's slider overrides actually put you.
"""

import math
from dataclasses import dataclass, field

from src.cost_modelling.calculator import (
    BYTES_PER_PARAM_CHECKPOINT,
    calculate_checkpoint_storage_tb,
    calculate_lora_trainable_params,
    calculate_storage_cost,
    calculate_training_flops,
    estimate_compute_cost,
    solve_for_parameter_count,
    solve_for_training_tokens,
)
from src.cost_modelling.gpu_specs import (
    get_gpu_instance,
    get_storage_cost,
    list_available_instances,
    peak_flops_for_precision,
)
from src.cost_modelling.constants import (
    CHINCHILLA_OPTIMAL_RATIO,
    LORA_OPTIMAL_RATIO,
)
from src.cost_modelling.model_loader import ModelDefinition

ARCH_MULTIPLIERS = {
    "Transformer": 6.0,
    "CNN": 4.0,
    "RNN": 8.0,
    "ViT": 6.0,
    "Diffusion": 6.5,
}

# bytes_per_token: average object-storage bytes per training token/sample
MODALITY_DEFAULTS = {
    "Text (LLM)": {
        "arch": "Transformer",
        "bytes_per_token": 4,
        "tokens_per_sample": 512,    # ~512 tokens per web document / article
        "sample_noun": "documents",
    },
    "Vision (ViT / CLIP)": {
        "arch": "ViT",
        "bytes_per_token": 1536,
        "tokens_per_sample": 196,    # 14×14 patches for 224×224 at 16×16 patch size
        "sample_noun": "images",
    },
    "Audio": {
        "arch": "Transformer",
        "bytes_per_token": 8,
        "tokens_per_sample": 750,    # 10-second clip at 75 tok/s (EnCodec / Whisper rate)
        "sample_noun": "clips",
    },
    "Multimodal (VLM)": {
        "arch": "Transformer",
        "bytes_per_token": 512,
        "tokens_per_sample": 512,    # mixed text + image patches per caption/pair
        "sample_noun": "samples",
    },
    "Diffusion (Image Gen)": {
        "arch": "Diffusion",
        "bytes_per_token": 2048,
        "tokens_per_sample": 1024,   # 32×32 latent patches for a 512×512 image (8× VAE)
        "sample_noun": "images",
    },
}

# Checkpoints saved per full training run (used for storage estimation)
CHECKPOINTS_PER_RUN = 5

# Target wall-clock days for auto-scaling num_instances
TARGET_WALL_CLOCK_DAYS = 60

# Hard cap on recommended instances — represents a realistic large-scale cluster
# (~8,192 H100 GPUs).  Wall-clock is reported honestly for whatever this delivers.
MAX_INSTANCES = 1_024

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Explore-direction labels. The available set depends on the training mode.
EXPLORE_DATASET = "Dataset size → token/param ratio"
EXPLORE_LORA_RANK = "LoRA Rank → dataset size"
EXPLORE_TOKENS_TO_MODEL = "Token count → model size"
EXPLORE_MODEL_TO_TOKENS = "Model size → token count"


def auto_configure_hardware() -> dict:
    """Pick the most cost-efficient available instance and return its config.

    Cost efficiency = effective_TFLOPS_per_$ = peak_flops × gpus × typical_mfu / hourly_cost.
    H100 (p5.48xlarge) currently wins: ~65 TFLOPS/$ vs A100 ~42 TFLOPS/$.
    """
    best_instance = None
    best_efficiency = -1.0

    for instance_type in list_available_instances():
        spec = get_gpu_instance(instance_type)
        flops_per_dollar = (
            spec["peak_flops_fp16"] * spec["gpu_count"] * spec["typical_mfu"]
            / spec["hourly_cost"]
        )
        if flops_per_dollar > best_efficiency:
            best_efficiency = flops_per_dollar
            best_instance = instance_type

    spec = get_gpu_instance(best_instance)
    peak_flops = peak_flops_for_precision(spec, "bf16")
    return {
        "instance_type": best_instance,
        "instance_spec": spec,
        "peak_flops_per_gpu": peak_flops,
        "mfu": spec["typical_mfu"],
        "gpus_per_instance": spec["gpu_count"],
        "hourly_cost": spec["hourly_cost"],
        "gradient_checkpointing": False,
    }


def quick_n_estimate(compute_budget: float, arch_multiplier: float, hw: dict) -> float:
    """Estimate model parameter count from budget alone (1-epoch Chinchilla, auto hardware).

    Used only for deriving sensible Training Schedule defaults — no storage correction needed.
    Approximate outputs at H100 pricing:
      $1K → ~140M params | $10K → ~450M | $100K → ~1.4B | $1M → ~14B | $10M → ~45B
    """
    quick_cost_per_tp = (
        arch_multiplier
        / (hw["peak_flops_per_gpu"] * hw["mfu"] * hw["gpus_per_instance"] * 3600)
    ) * hw["hourly_cost"]
    if quick_cost_per_tp <= 0:
        return 0.0
    return math.sqrt(compute_budget / (CHINCHILLA_OPTIMAL_RATIO * quick_cost_per_tp))


def derive_schedule_defaults(
    compute_budget: float,
    modality: str,
    training_type: str,
    ft_method: str | None,
    quick_n: float,
) -> dict:
    """Derive recommended Training Schedule & Storage defaults from project-setup inputs.

    Returns a dict with keys: epochs, hp_trials, hp_fraction_pct, storage_months, storage_class.
    """
    is_lora = ft_method in ("LoRA", "QLoRA")

    # Epochs — driven by training type
    # Pre-training is 1-epoch optimal under Chinchilla; FT benefits from more passes.
    if training_type == "Pre-Training":
        epochs = 1
    elif is_lora:
        epochs = 5
    else:
        epochs = 3

    # HP Tuning Trials — scaled with budget (more budget = can afford more HP search)
    if compute_budget < 5_000:
        hp_trials = 0
    elif compute_budget < 50_000:
        hp_trials = 2
    elif compute_budget < 500_000:
        hp_trials = 5
    elif compute_budget < 5_000_000:
        hp_trials = 10
    else:
        hp_trials = 20

    # HP Trial Cost % — larger models need shorter (cheaper) HP trials
    if quick_n >= 30e9:
        hp_fraction_pct = 5
    elif quick_n >= 3e9:
        hp_fraction_pct = 10
    elif quick_n >= 500e6:
        hp_fraction_pct = 15
    else:
        hp_fraction_pct = 25

    # Storage Duration — bigger budget = more valuable artifacts = keep longer
    # Vision/Diffusion have expensive storage so halve the duration
    bytes_per_token = MODALITY_DEFAULTS[modality]["bytes_per_token"]
    storage_heavy = bytes_per_token > 100  # Vision, Multimodal, Diffusion

    if compute_budget < 10_000:
        base_months = 1
    elif compute_budget < 100_000:
        base_months = 3
    elif compute_budget < 1_000_000:
        base_months = 6
    else:
        base_months = 12

    storage_months = max(1, base_months // 2 if storage_heavy else base_months)

    # Storage Class — longer retention → cheaper tier
    storage_class = "standard_ia" if storage_months > 3 else "standard"

    return {
        "epochs": epochs,
        "hp_trials": hp_trials,
        "hp_fraction_pct": hp_fraction_pct,
        "storage_months": storage_months,
        "storage_class": storage_class,
    }


def budget_curve_n(
    d_tokens: list[float],
    budget: float,
    peak_flops_per_gpu: float,
    mfu: float,
    gpus_per_instance: int,
    hourly_cost: float,
    multiplier: float,
    epochs: int,
) -> list[float]:
    """N_max = budget / (cost_per_token_param × D), evaluated across a token axis.

    num_instances cancels in the cost formula, so we pass gpus_per_instance directly.
    """
    cost_per_token_param = (
        epochs * multiplier / (peak_flops_per_gpu * mfu * gpus_per_instance * 3600)
    ) * hourly_cost
    return [budget / (cost_per_token_param * d) for d in d_tokens]


def logspace(start: float, stop: float, num: int) -> list[float]:
    """Stand-in for ``numpy.logspace`` — evenly spaced on a log10 scale, endpoint included."""
    if num < 2:
        return [10.0 ** start] * max(num, 0)
    step = (stop - start) / (num - 1)
    return [10.0 ** (start + i * step) for i in range(num)]


@dataclass
class OptimizerInputs:
    """Everything the solve needs. All of it comes straight from Project Setup widgets."""

    compute_budget: float
    modality: str
    training_type: str
    ft_method: str | None = None
    # Resolved base model, or None for pre-training / "Custom".
    ft_base_model: ModelDefinition | None = None
    lora_rank: int = 16
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    epochs: int = 1
    num_hp_trials: int = 0
    hp_fraction_pct: int = 10
    storage_duration_months: int = 3
    storage_class: str = "standard"
    hardware: dict | None = None

    def __post_init__(self) -> None:
        if self.hardware is None:
            self.hardware = auto_configure_hardware()

    @property
    def hw(self) -> dict:
        return self.hardware

    @property
    def is_lora(self) -> bool:
        return self.ft_method in ("LoRA", "QLoRA")

    @property
    def is_ft_with_base(self) -> bool:
        return self.training_type == "Fine-Tuning" and self.ft_base_model is not None

    @property
    def modality_cfg(self) -> dict:
        return MODALITY_DEFAULTS[self.modality]

    @property
    def architecture(self) -> str:
        return self.modality_cfg["arch"]

    @property
    def bytes_per_token(self) -> int:
        return self.modality_cfg["bytes_per_token"]

    @property
    def multiplier(self) -> float:
        return ARCH_MULTIPLIERS.get(self.architecture, 6.0)

    @property
    def base_params_total(self) -> int:
        """Total params — checkpoints, VRAM, adapter percentages."""
        return self.ft_base_model.parameter_count if self.is_ft_with_base else 0

    @property
    def base_params_active(self) -> int:
        """Active params — compute only. Differs from the total for MoE models."""
        return self.ft_base_model.effective_parameter_count if self.is_ft_with_base else 0

    def adapter_params(self, lora_rank: int | None = None) -> int:
        """Trainable / displayed parameter count for fine-tuning."""
        if not self.is_ft_with_base:
            return 0
        if not self.is_lora:
            return self.base_params_total   # SFT: all params are trainable
        return calculate_lora_trainable_params(
            ft_method=self.ft_method,
            base_params=self.base_params_total,
            target_modules=self.target_modules or [],
            d_model=self.ft_base_model.architecture.get("d_model", 0),
            num_layers=self.ft_base_model.architecture.get("num_layers", 0),
            lora_rank=self.lora_rank if lora_rank is None else lora_rank,
            architecture=self.ft_base_model.architecture,
        ) or 0

    @property
    def explore_options(self) -> list[str]:
        """Explore directions valid for the current mode.

        LoRA/QLoRA: dataset size (D varies, adapter N fixed) or rank (N and D both change).
        SFT: dataset size only (N is fixed at the base model).
        Pre-training: free solve in either direction.
        """
        if self.is_ft_with_base and self.is_lora:
            return [EXPLORE_DATASET, EXPLORE_LORA_RANK]
        if self.is_ft_with_base:
            return [EXPLORE_DATASET]
        return [EXPLORE_TOKENS_TO_MODEL, EXPLORE_MODEL_TO_TOKENS]


@dataclass
class BudgetOptimum:
    """The scaling-law-optimal point for a budget, plus the economics used to get there."""

    n_opt: float
    d_opt: float
    n_adapter: int
    n_flops_base: int
    # The two competing LoRA bounds on D. Both 0.0 outside the LoRA-with-base path;
    # the page only surfaces them when is_ft_with_base and is_lora.
    d_eff: float
    d_budget: float
    opt_compute_cost_total: float
    opt_dataset_storage_cost: float
    opt_ckpt_storage_cost: float
    opt_storage_cost: float
    total_project_cost: float
    # Shared economics — resolve_selection reuses these rather than recomputing them.
    project_multiplier: float
    cost_per_token_param: float
    ckpt_cost_per_param: float
    storage_cost_per_token: float
    cost_per_tb_month: float
    optimal_ratio: int


def solve_budget_optimum(inputs: OptimizerInputs) -> BudgetOptimum:
    """Solve for the optimal (n_opt, d_opt) within the budget, and cost it out."""
    hw = inputs.hw
    peak_flops_per_gpu = hw["peak_flops_per_gpu"]
    mfu = hw["mfu"]
    gpus_per_instance = hw["gpus_per_instance"]
    hourly_cost = hw["hourly_cost"]
    gradient_checkpointing = hw["gradient_checkpointing"]

    compute_budget = inputs.compute_budget
    bytes_per_token = inputs.bytes_per_token
    architecture = inputs.architecture
    epochs = inputs.epochs
    num_hp_trials = inputs.num_hp_trials
    storage_duration_months = inputs.storage_duration_months
    storage_class = inputs.storage_class
    is_lora = inputs.is_lora
    is_ft_with_base = inputs.is_ft_with_base
    n_adapter = inputs.adapter_params()

    hp_fraction = inputs.hp_fraction_pct / 100.0
    project_multiplier = 1.0 + num_hp_trials * hp_fraction
    cost_per_tb_month = get_storage_cost(storage_class)
    storage_cost_per_token = (
        (bytes_per_token / 1e12) * cost_per_tb_month * storage_duration_months
    )

    optimal_ratio = LORA_OPTIMAL_RATIO if is_lora else CHINCHILLA_OPTIMAL_RATIO

    # cost per (parameter × token) — num_instances cancels in cost formula
    cost_per_token_param = (
        epochs * inputs.multiplier / (peak_flops_per_gpu * mfu * gpus_per_instance * 3600)
    ) * hourly_cost

    ckpt_cost_per_param = (
        BYTES_PER_PARAM_CHECKPOINT / 1e12
        * cost_per_tb_month
        * storage_duration_months
        * (CHECKPOINTS_PER_RUN + num_hp_trials)
    )

    if is_ft_with_base and n_adapter > 0:
        # Fine-tuning with a known base model: N is fixed by the model + rank.
        # For LoRA/QLoRA the forward/backward pass runs through the full base model,
        # so compute cost uses the base params; adapter N is used for checkpoints only.
        #
        # Two competing bounds on D:
        #
        # 1) LoRA efficiency bound — empirical optimal ratio (D/N_adapter ≈ 50):
        #      D_eff = LORA_OPTIMAL_RATIO × n_adapter
        #    This scales with rank and is the primary bound at large budgets.
        #
        # 2) Budget bound — how much data the full budget can afford:
        #      D_budget = (B − ckpt_cost) / (cost_per_token_param × N_base + storage_per_token)
        #    This is the hard ceiling; at small budgets it overrides the efficiency bound.
        #
        # D_opt = min(D_eff, D_budget)
        n_flops_base = inputs.base_params_active   # used for compute FLOPs
        n_opt = float(n_adapter)                   # displayed / checkpoint param count
        ckpt_total = ckpt_cost_per_param * n_adapter
        denom = (
            project_multiplier * cost_per_token_param * n_flops_base
            + storage_cost_per_token
        )
        d_budget = max(compute_budget - ckpt_total, 0.0) / denom if denom > 0 else 0.0
        d_eff = float(LORA_OPTIMAL_RATIO * n_adapter) if is_lora else d_budget
        d_opt = min(d_eff, d_budget)
    else:
        # Pre-training (or FT without a selected base model): quadratic solve.
        # Total budget B = compute_total + dataset_storage + checkpoint_storage
        # Under Chinchilla D = optimal_ratio × N:  a·N² + b·N = B
        #   a = optimal_ratio × project_multiplier × cost_per_token_param
        #   b = optimal_ratio × storage_cost_per_token + ckpt_cost_per_param
        # N* = (−b + √(b² + 4aB)) / (2a)
        n_flops_base = 0   # not used in pre-training path
        d_eff = d_budget = 0.0   # LoRA-only bounds
        a_coeff = optimal_ratio * project_multiplier * cost_per_token_param
        b_coeff = optimal_ratio * storage_cost_per_token + ckpt_cost_per_param
        if a_coeff > 0:
            discriminant = b_coeff ** 2 + 4 * a_coeff * compute_budget
            n_opt = (-b_coeff + math.sqrt(max(discriminant, 0.0))) / (2 * a_coeff)
            d_opt = optimal_ratio * n_opt
        else:
            n_opt, d_opt = 0.0, 0.0

    # ── Cost breakdown at optimal point ──────────────────────────────────────
    # Use the base model's active params for FLOPs when fine-tuning (forward pass runs
    # through the full model); use n_opt (adapter params or pre-train N) for checkpoints.
    _opt_flops_n = max(
        int(inputs.base_params_active if (is_ft_with_base and n_flops_base) else n_opt), 1
    )
    opt_flops = calculate_training_flops(
        parameter_count=_opt_flops_n,
        training_tokens=max(int(d_opt), 1),
        architecture=architecture.lower(),
        epochs=epochs,
        gradient_checkpointing=gradient_checkpointing,
    )
    _, _, opt_compute_cost_per_run, _ = estimate_compute_cost(
        total_flops=opt_flops,
        peak_flops_per_gpu=peak_flops_per_gpu,
        mfu=mfu,
        total_gpus=gpus_per_instance,
        num_instances=1,
        hourly_cost=hourly_cost,
    )
    opt_compute_cost_total = opt_compute_cost_per_run * project_multiplier
    opt_dataset_storage_cost = calculate_storage_cost(
        dataset_size_tb=max(int(d_opt), 1) * bytes_per_token / 1e12,
        storage_duration_months=storage_duration_months,
        storage_class=storage_class,
    )
    opt_ckpt_storage_cost = calculate_storage_cost(
        dataset_size_tb=calculate_checkpoint_storage_tb(
            checkpoint_params=max(int(n_opt), 1),
            num_checkpoints=CHECKPOINTS_PER_RUN,
            num_training_runs=1,
            num_hp_trials=num_hp_trials,
            num_ablations=0,
        ),
        storage_duration_months=storage_duration_months,
        storage_class=storage_class,
    )
    opt_storage_cost = opt_dataset_storage_cost + opt_ckpt_storage_cost
    total_project_cost = opt_compute_cost_total + opt_storage_cost

    return BudgetOptimum(
        n_opt=n_opt,
        d_opt=d_opt,
        n_adapter=n_adapter,
        n_flops_base=n_flops_base,
        d_eff=d_eff,
        d_budget=d_budget,
        opt_compute_cost_total=opt_compute_cost_total,
        opt_dataset_storage_cost=opt_dataset_storage_cost,
        opt_ckpt_storage_cost=opt_ckpt_storage_cost,
        opt_storage_cost=opt_storage_cost,
        total_project_cost=total_project_cost,
        project_multiplier=project_multiplier,
        cost_per_token_param=cost_per_token_param,
        ckpt_cost_per_param=ckpt_cost_per_param,
        storage_cost_per_token=storage_cost_per_token,
        cost_per_tb_month=cost_per_tb_month,
        optimal_ratio=optimal_ratio,
    )


@dataclass
class Selection:
    """User overrides from the trade-off explorer.

    ``None`` means "not overridden" — the caller should fall back to the derived default.
    This mirrors Streamlit's ``session_state.pop(key)``: absence, not a written-in default.
    """

    explore_dir: str | None = None
    log_d: float | None = None
    log_n: float | None = None
    rank: int | None = None
    num_instances: int | None = None


@dataclass
class SelectionResult:
    """Everything the page renders once the user's overrides are applied."""

    explore_dir: str
    explore_options: list[str]
    selected_tokens: int
    selected_params: int
    flops_params: int
    n_axis_title: str
    n_hover: str
    total_flops_sel: float
    compute_cost_sel: float
    wall_clock_days_sel: float
    wall_clock_hours_1inst: float
    recommended_instances: int
    num_instances: int
    sel_dataset_storage: float
    sel_ckpt_storage: float
    sel_compute_cost_total: float
    sel_total_cost: float
    sel_wall_clock_days: float
    sel_samples: float
    current_ratio: float
    budget_used_pct: float
    # Slider geometry — the page needs these to render bounds and defaults.
    d_slider_min: float
    d_slider_max: float
    n_slider_min: float
    n_slider_max: float
    d_opt_log_default: float
    n_opt_log_default: float
    slider_effective_budget: float
    tokens_per_sample: int
    sample_noun: str


def resolve_selection(
    inputs: OptimizerInputs,
    optimum: BudgetOptimum,
    selection: Selection | None = None,
) -> SelectionResult:
    """Apply the user's trade-off overrides on top of the optimal point."""
    selection = selection or Selection()

    hw = inputs.hw
    peak_flops_per_gpu = hw["peak_flops_per_gpu"]
    mfu = hw["mfu"]
    gpus_per_instance = hw["gpus_per_instance"]
    hourly_cost = hw["hourly_cost"]
    gradient_checkpointing = hw["gradient_checkpointing"]

    compute_budget = inputs.compute_budget
    bytes_per_token = inputs.bytes_per_token
    architecture = inputs.architecture
    epochs = inputs.epochs
    num_hp_trials = inputs.num_hp_trials
    storage_duration_months = inputs.storage_duration_months
    storage_class = inputs.storage_class
    is_lora = inputs.is_lora
    is_ft_with_base = inputs.is_ft_with_base

    d_opt = optimum.d_opt
    n_opt = optimum.n_opt
    n_adapter = optimum.n_adapter
    project_multiplier = optimum.project_multiplier

    modality_cfg = inputs.modality_cfg
    tokens_per_sample = modality_cfg.get("tokens_per_sample", 512)
    sample_noun = modality_cfg.get("sample_noun", "samples")

    slider_compute_budget = max(
        compute_budget - optimum.opt_storage_cost, compute_budget * 0.5
    )
    slider_effective_budget = slider_compute_budget / project_multiplier

    # Dynamic slider bounds — scale to the optimal point.
    # For LoRA, d_opt can be well below 1B tokens (e.g. 50 × a few-million adapter params),
    # so we anchor the minimum ~2 log-decades below d_opt instead of hardcoding 1B.
    if is_ft_with_base and is_lora and d_opt > 0:
        d_slider_min = max(6.0, math.floor(math.log10(max(d_opt, 1e6)) / 0.05) * 0.05 - 2.0)
    else:
        d_slider_min = 9.0   # pre-training / SFT: 1B token minimum

    d_log_opt = math.log10(max(d_opt, 10 ** d_slider_min))
    d_slider_max = max(d_slider_min + 4.0, math.ceil(d_log_opt / 0.05) * 0.05 + 0.05)

    n_log_opt = math.log10(max(n_opt, 1e7))
    n_slider_min = 7.0
    n_slider_max = max(11.0, math.ceil(n_log_opt / 0.05) * 0.05 + 0.05)

    # Slider defaults — identical expressions in every branch of the original, hoisted.
    d_opt_log_raw = math.log10(max(d_opt, 10 ** d_slider_min))
    d_opt_log_default = (
        round(max(d_slider_min, min(d_slider_max, d_opt_log_raw)) / 0.05) * 0.05
    )
    n_opt_log_raw = math.log10(max(n_opt, 1e7))
    n_opt_log_default = (
        round(max(n_slider_min, min(n_slider_max, n_opt_log_raw)) / 0.05) * 0.05
    )

    explore_options = inputs.explore_options
    explore_dir = selection.explore_dir
    # Guard: if the stored direction isn't valid for the current mode, reset it.
    if explore_dir not in explore_options:
        explore_dir = explore_options[0]

    if is_ft_with_base and is_lora:
        # LoRA mode: N is determined by base model + rank.  Both slider directions are useful.
        if explore_dir == EXPLORE_LORA_RANK:
            # Rank slider: user picks rank, we solve for optimal D.
            sel_rank = int(selection.rank if selection.rank is not None else inputs.lora_rank)
            sel_n_adapter = inputs.adapter_params(lora_rank=sel_rank) or 1
            # Linear solve for D with this hypothetical rank
            _ckpt_n = sel_n_adapter
            _denom = (
                project_multiplier * optimum.cost_per_token_param * inputs.base_params_active
                + optimum.storage_cost_per_token
            )
            _eff_bgt = max(compute_budget - optimum.ckpt_cost_per_param * _ckpt_n, 0.0)
            selected_tokens = max(int(_eff_bgt / _denom), 1) if _denom > 0 else 1
            selected_params = sel_n_adapter
        else:
            # Dataset slider: D varies, adapter N stays fixed at Project Setup rank.
            log_d = selection.log_d if selection.log_d is not None else d_opt_log_default
            selected_tokens = int(10 ** log_d)
            selected_params = max(n_adapter, 1)
        # FLOPs: always use full base model (frozen forward/backward pass)
        flops_params = inputs.base_params_active
        n_axis_title = "Adapter params"
        n_hover = "Adapter N"

    elif is_ft_with_base:
        # SFT: all params are trained.  Only dataset exploration makes sense.
        log_d = selection.log_d if selection.log_d is not None else d_opt_log_default
        selected_tokens = int(10 ** log_d)
        selected_params = inputs.base_params_total
        flops_params = inputs.base_params_active
        n_axis_title = "Model size (params)"
        n_hover = "N"

    else:
        # Pre-training: free solve in both directions.
        if explore_dir == EXPLORE_TOKENS_TO_MODEL:
            log_d = selection.log_d if selection.log_d is not None else d_opt_log_default
            selected_tokens = int(10 ** log_d)
            selected_params = solve_for_parameter_count(
                compute_budget_usd=slider_effective_budget,
                training_tokens=selected_tokens,
                peak_flops_per_gpu=peak_flops_per_gpu,
                mfu=mfu,
                total_gpus=gpus_per_instance,
                num_instances=1,
                hourly_cost=hourly_cost,
                architecture=architecture.lower(),
                epochs=epochs,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            log_n = selection.log_n if selection.log_n is not None else n_opt_log_default
            selected_params = int(10 ** log_n)
            selected_tokens = solve_for_training_tokens(
                compute_budget_usd=slider_effective_budget,
                parameter_count=selected_params,
                peak_flops_per_gpu=peak_flops_per_gpu,
                mfu=mfu,
                total_gpus=gpus_per_instance,
                num_instances=1,
                hourly_cost=hourly_cost,
                architecture=architecture.lower(),
                epochs=epochs,
                gradient_checkpointing=gradient_checkpointing,
            )
        flops_params = selected_params
        n_axis_title = "Max model size (params)"
        n_hover = "Max N"

    total_flops_sel = calculate_training_flops(
        parameter_count=max(flops_params, 1),
        training_tokens=max(selected_tokens, 1),
        architecture=architecture.lower(),
        epochs=epochs,
        gradient_checkpointing=gradient_checkpointing,
    )
    _, _, compute_cost_sel, wall_clock_days_sel = estimate_compute_cost(
        total_flops=total_flops_sel,
        peak_flops_per_gpu=peak_flops_per_gpu,
        mfu=mfu,
        total_gpus=gpus_per_instance,
        num_instances=1,
        hourly_cost=hourly_cost,
    )

    # Recommend num_instances to hit ≤ TARGET_WALL_CLOCK_DAYS for the *selected* point.
    # Deriving it here (after the slider solve) ensures the hardware card and the
    # wall-clock metric are always derived from the same FLOP count.
    wall_clock_hours_1inst = total_flops_sel / (
        peak_flops_per_gpu * mfu * gpus_per_instance * 3600
    )
    recommended_instances = min(
        MAX_INSTANCES,
        max(1, math.ceil(wall_clock_hours_1inst / (TARGET_WALL_CLOCK_DAYS * 24))),
    )
    # User can override the cluster size; default resets whenever setup changes.
    num_instances = (
        selection.num_instances if selection.num_instances is not None
        else recommended_instances
    )

    sel_dataset_storage = calculate_storage_cost(
        dataset_size_tb=max(selected_tokens, 1) * bytes_per_token / 1e12,
        storage_duration_months=storage_duration_months,
        storage_class=storage_class,
    )
    sel_ckpt_storage = calculate_storage_cost(
        dataset_size_tb=calculate_checkpoint_storage_tb(
            checkpoint_params=max(selected_params, 1),
            num_checkpoints=CHECKPOINTS_PER_RUN,
            num_training_runs=1,
            num_hp_trials=num_hp_trials,
            num_ablations=0,
        ),
        storage_duration_months=storage_duration_months,
        storage_class=storage_class,
    )
    sel_compute_cost_total = compute_cost_sel * project_multiplier
    sel_total_cost = sel_compute_cost_total + sel_dataset_storage + sel_ckpt_storage
    sel_wall_clock_days = wall_clock_hours_1inst / (num_instances * 24)
    sel_samples = selected_tokens / max(tokens_per_sample, 1)
    current_ratio = selected_tokens / max(selected_params, 1)
    budget_used_pct = min(sel_total_cost / max(compute_budget, 1e-9), 1.0)

    return SelectionResult(
        explore_dir=explore_dir,
        explore_options=explore_options,
        selected_tokens=selected_tokens,
        selected_params=selected_params,
        flops_params=flops_params,
        n_axis_title=n_axis_title,
        n_hover=n_hover,
        total_flops_sel=total_flops_sel,
        compute_cost_sel=compute_cost_sel,
        wall_clock_days_sel=wall_clock_days_sel,
        wall_clock_hours_1inst=wall_clock_hours_1inst,
        recommended_instances=recommended_instances,
        num_instances=num_instances,
        sel_dataset_storage=sel_dataset_storage,
        sel_ckpt_storage=sel_ckpt_storage,
        sel_compute_cost_total=sel_compute_cost_total,
        sel_total_cost=sel_total_cost,
        sel_wall_clock_days=sel_wall_clock_days,
        sel_samples=sel_samples,
        current_ratio=current_ratio,
        budget_used_pct=budget_used_pct,
        d_slider_min=d_slider_min,
        d_slider_max=d_slider_max,
        n_slider_min=n_slider_min,
        n_slider_max=n_slider_max,
        d_opt_log_default=d_opt_log_default,
        n_opt_log_default=n_opt_log_default,
        slider_effective_budget=slider_effective_budget,
        tokens_per_sample=tokens_per_sample,
        sample_noun=sample_noun,
    )
