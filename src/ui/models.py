from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AppSettings:
    """Aggregated configuration returned from the sidebar."""
    azure_api_key: Optional[str]
    azure_endpoint: Optional[str]
    azure_api_version: Optional[str]
    azure_model_name: Optional[str]
    asr_model_info: str
    use_local_llm: bool
    local_llm_model: str
    use_encryption: bool
    language: Optional[str]
    use_agent_pipeline: bool

@dataclass
class AgentSettings:
    """Settings specific to the agent-based pipeline."""
    max_refinements: int
    enable_stage_display: bool
    quality_threshold: int

    def as_dict(self) -> Dict[str, Any]:
        """Helper to convert settings to a plain dictionary."""
        return {
            "max_refinements": self.max_refinements,
            "enable_stage_display": self.enable_stage_display,
            "quality_threshold": self.quality_threshold,
        }
