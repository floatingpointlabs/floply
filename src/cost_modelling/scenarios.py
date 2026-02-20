"""Predefined training scenarios for quick cost estimation.

This module contains pre-configured scenarios representing common training workloads:
1. Quick Validation - Test training loop
2. Small-Scale Research - Proof of concept
3. Emerging Capabilities - See emergent behavior
4. Production Training - Full-scale model
"""

from typing import Dict, List
from src.cost_modelling.calculator import ModelConfig, TrainingConfig, CostBreakdown, calculate_single_run_cost


class Scenario:
    """A predefined training scenario."""
    
    def __init__(
        self,
        name: str,
        description: str,
        goal: str,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        tags: List[str] = None
    ):
        self.name = name
        self.description = description
        self.goal = goal
        self.model_config = model_config
        self.training_config = training_config
        self.tags = tags or []
        self._cost_breakdown: CostBreakdown | None = None
    
    @property
    def cost_breakdown(self) -> CostBreakdown:
        """Calculate and cache cost breakdown."""
        if self._cost_breakdown is None:
            self._cost_breakdown = calculate_single_run_cost(
                self.model_config,
                self.training_config
            )
        return self._cost_breakdown
    
    def to_dict(self) -> Dict:
        """Convert scenario to dictionary."""
        cost = self.cost_breakdown
        return {
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "model_size": self.model_config.to_human_readable(),
            "training_tokens": f"{self.training_config.training_tokens / 1e9:.1f}B" if self.training_config.training_tokens >= 1e9 else f"{self.training_config.training_tokens / 1e6:.0f}M",
            "instance": f"{self.training_config.num_instances}x {self.training_config.instance_type}",
            "gpu_type": cost.gpu_type,
            "total_gpus": cost.total_gpus,
            "duration_days": f"{cost.wall_clock_days:.1f}",
            "duration_hours": f"{cost.wall_clock_hours:.1f}",
            "total_cost": f"${cost.total_cost:,.2f}",
            "cost_numeric": cost.total_cost,
            "gpu_hours": f"{cost.gpu_hours:,.0f}",
            "tags": self.tags,
        }


# Scenario 1: Quick Validation
QUICK_VALIDATION = Scenario(
    name="Quick Validation",
    description="Test training loop, verify loss decreases",
    goal="Validate that the training pipeline works and the model can learn",
    model_config=ModelConfig(
        parameter_count=100_000_000,  # 100M parameters
        architecture="transformer",
    ),
    training_config=TrainingConfig(
        training_tokens=1_000_000_000,  # 1B tokens
        batch_size=512,
        gradient_accumulation_steps=1,
        epochs=1.0,
        mixed_precision="fp16",
        instance_type="p4d.24xlarge",
        num_instances=1,
        mfu_override=0.40,  # Lower MFU for quick test
    ),
    tags=["validation", "quick", "debugging", "small"]
)

# Scenario 2: Small-Scale Research
SMALL_SCALE_RESEARCH = Scenario(
    name="Small-Scale Research",
    description="Proof of concept, early experiments",
    goal="Conduct initial research experiments and validate hypotheses",
    model_config=ModelConfig(
        parameter_count=350_000_000,  # 350M parameters
        architecture="transformer",
    ),
    training_config=TrainingConfig(
        training_tokens=50_000_000_000,  # 50B tokens
        batch_size=1024,
        gradient_accumulation_steps=2,
        epochs=1.0,
        mixed_precision="fp16",
        instance_type="p4d.24xlarge",
        num_instances=1,
        mfu_override=0.45,
    ),
    tags=["research", "poc", "experiment", "medium"]
)

# Scenario 3: Emerging Capabilities
EMERGING_CAPABILITIES = Scenario(
    name="Emerging Capabilities",
    description="See emergent behavior, validate approach at scale",
    goal="Train a model large enough to exhibit emergent capabilities",
    model_config=ModelConfig(
        parameter_count=1_000_000_000,  # 1B parameters
        architecture="transformer",
    ),
    training_config=TrainingConfig(
        training_tokens=200_000_000_000,  # 200B tokens
        batch_size=2048,
        gradient_accumulation_steps=4,
        epochs=1.0,
        mixed_precision="bf16",
        instance_type="p4d.24xlarge",
        num_instances=1,
        mfu_override=0.50,
    ),
    tags=["research", "emergent", "large", "validation"]
)

# Scenario 4: Production Training
PRODUCTION_TRAINING = Scenario(
    name="Production Training",
    description="Full-scale model for deployment",
    goal="Train a production-ready model with state-of-the-art performance",
    model_config=ModelConfig(
        parameter_count=7_000_000_000,  # 7B parameters
        architecture="transformer",
    ),
    training_config=TrainingConfig(
        training_tokens=1_000_000_000_000,  # 1T tokens
        batch_size=4096,
        gradient_accumulation_steps=8,
        epochs=1.0,
        mixed_precision="bf16",
        instance_type="p5.48xlarge",
        num_instances=8,
        mfu_override=0.55,
    ),
    tags=["production", "large", "deployment", "sota"]
)

# Additional Scenario: Medium Research Model
MEDIUM_RESEARCH = Scenario(
    name="Medium Research Model",
    description="Mid-size model for advanced research",
    goal="Train a 3B parameter model for in-depth research",
    model_config=ModelConfig(
        parameter_count=3_000_000_000,  # 3B parameters
        architecture="transformer",
    ),
    training_config=TrainingConfig(
        training_tokens=500_000_000_000,  # 500B tokens
        batch_size=2048,
        gradient_accumulation_steps=4,
        epochs=1.0,
        mixed_precision="bf16",
        instance_type="p4d.24xlarge",
        num_instances=4,
        mfu_override=0.48,
    ),
    tags=["research", "medium", "advanced"]
)

# Additional Scenario: Large-Scale Training (13B)
LARGE_SCALE_TRAINING = Scenario(
    name="Large-Scale Training (13B)",
    description="Large 13B parameter model",
    goal="Train a large-scale model for advanced applications",
    model_config=ModelConfig(
        parameter_count=13_000_000_000,  # 13B parameters
        architecture="transformer",
    ),
    training_config=TrainingConfig(
        training_tokens=1_300_000_000_000,  # 1.3T tokens
        batch_size=4096,
        gradient_accumulation_steps=8,
        epochs=1.0,
        mixed_precision="bf16",
        instance_type="p5.48xlarge",
        num_instances=16,
        mfu_override=0.56,
    ),
    tags=["production", "large-scale", "advanced"]
)


# Dictionary of all scenarios
ALL_SCENARIOS: Dict[str, Scenario] = {
    "quick_validation": QUICK_VALIDATION,
    "small_scale_research": SMALL_SCALE_RESEARCH,
    "emerging_capabilities": EMERGING_CAPABILITIES,
    "production_training": PRODUCTION_TRAINING,
    "medium_research": MEDIUM_RESEARCH,
    "large_scale_training": LARGE_SCALE_TRAINING,
}


def get_scenario(scenario_name: str) -> Scenario:
    """Get a scenario by name.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        Scenario object
        
    Raises:
        ValueError: If scenario not found
    """
    if scenario_name not in ALL_SCENARIOS:
        available = ", ".join(ALL_SCENARIOS.keys())
        raise ValueError(
            f"Unknown scenario: {scenario_name}. "
            f"Available scenarios: {available}"
        )
    return ALL_SCENARIOS[scenario_name]


def list_scenarios() -> List[str]:
    """List all available scenario names."""
    return list(ALL_SCENARIOS.keys())


def get_scenarios_by_tag(tag: str) -> List[Scenario]:
    """Get all scenarios with a specific tag.
    
    Args:
        tag: Tag to filter by
        
    Returns:
        List of matching scenarios
    """
    return [
        scenario for scenario in ALL_SCENARIOS.values()
        if tag in scenario.tags
    ]


def compare_scenarios(scenario_names: List[str] = None) -> List[Dict]:
    """Compare multiple scenarios.
    
    Args:
        scenario_names: List of scenario names to compare (None = all)
        
    Returns:
        List of scenario dictionaries for comparison
    """
    if scenario_names is None:
        scenarios_to_compare = list(ALL_SCENARIOS.values())
    else:
        scenarios_to_compare = [get_scenario(name) for name in scenario_names]
    
    return [scenario.to_dict() for scenario in scenarios_to_compare]


def get_recommended_scenario(
    budget: float,
    timeline_days: float = None,
    goal: str = "research"
) -> Scenario:
    """Recommend a scenario based on constraints.
    
    Args:
        budget: Maximum budget in USD
        timeline_days: Maximum timeline in days (optional)
        goal: Goal type ('validation', 'research', 'production')
        
    Returns:
        Recommended scenario
    """
    # Filter by goal
    goal_tags = {
        "validation": ["validation", "debugging"],
        "research": ["research", "experiment"],
        "production": ["production", "deployment"],
    }
    
    relevant_tags = goal_tags.get(goal.lower(), ["research"])
    candidates = []
    
    for scenario in ALL_SCENARIOS.values():
        if any(tag in scenario.tags for tag in relevant_tags):
            cost = scenario.cost_breakdown
            if cost.total_cost <= budget:
                if timeline_days is None or cost.wall_clock_days <= timeline_days:
                    candidates.append(scenario)
    
    if not candidates:
        # Return cheapest scenario if no match
        return min(ALL_SCENARIOS.values(), key=lambda s: s.cost_breakdown.total_cost)
    
    # Return most expensive scenario within budget (maximize value)
    return max(candidates, key=lambda s: s.cost_breakdown.total_cost)
