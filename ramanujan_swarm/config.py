"""Configuration management for Ramanujan-Swarm."""

import os
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Config(BaseModel):
    """Global configuration."""

    # LLM Provider Selection
    llm_provider: str = os.getenv("LLM_PROVIDER", "claude").lower()  # "claude", "gemini", "bedrock", or "blackbox"

    # API Keys
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    blackbox_api_key: str = os.getenv("BLACKBOX_API_KEY", "")

    # AWS Bedrock (supports both bearer token and access keys)
    aws_bearer_token: str = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")

    # Swarm parameters
    swarm_size: int = int(os.getenv("SWARM_SIZE", "10"))
    max_generations: int = int(os.getenv("MAX_GENERATIONS", "20"))
    gene_pool_size: int = int(os.getenv("GENE_POOL_SIZE", "15"))

    # Mathematical constants to target
    target_constants: List[str] = os.getenv(
        "TARGET_CONSTANTS", "pi,e,phi"
    ).split(",")

    # Precision (decimal places)
    precision_dps: int = int(os.getenv("PRECISION_DPS", "100"))

    # Thresholds
    evolution_threshold: float = 1e-12  # Keep for evolution
    discovery_threshold: float = 1e-50  # Mark as discovery

    # LLM settings (provider-specific)
    def _get_default_model():
        provider = os.getenv("LLM_PROVIDER", "claude").lower()
        if provider == "claude":
            return "claude-3-5-sonnet-20240620"
        elif provider == "gemini":
            return "gemini-2.0-flash-exp"
        elif provider == "bedrock":
            # Use EU cross-region inference profile for Claude Sonnet 4.5
            # Available profiles: eu., us., jp., etc.
            return "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
        elif provider == "blackbox":
            # Blackbox AI model format: blackboxai/provider/model
            return "blackboxai/anthropic/claude-3.5-sonnet"
        return "claude-3-5-sonnet-20240620"

    llm_model: str = os.getenv("LLM_MODEL", _get_default_model())
    llm_temperature: float = 1.0  # Bedrock max is 1.0, others allow >1.0
    llm_max_tokens: int = 4096

    # Agent distribution (must sum to swarm_size or be fractions)
    explorer_fraction: float = 0.4
    mutator_fraction: float = 0.4
    hybrid_fraction: float = 0.2

    def validate_api_keys(self) -> bool:
        """Validate that the appropriate API key is set."""
        if self.llm_provider == "claude":
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY must be set when using Claude")
            return True
        elif self.llm_provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY must be set when using Gemini")
            return True
        elif self.llm_provider == "bedrock":
            # Bedrock supports both bearer token and access key authentication
            has_bearer = bool(self.aws_bearer_token)
            has_keys = bool(self.aws_access_key_id and self.aws_secret_access_key)
            if not has_bearer and not has_keys:
                raise ValueError(
                    "AWS Bedrock requires either AWS_BEARER_TOKEN_BEDROCK or "
                    "(AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)"
                )
            return True
        elif self.llm_provider == "blackbox":
            if not self.blackbox_api_key:
                raise ValueError("BLACKBOX_API_KEY must be set when using Blackbox AI")
            return True
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}. Use 'claude', 'gemini', 'bedrock', or 'blackbox'.")


# Global config instance
config = Config()
