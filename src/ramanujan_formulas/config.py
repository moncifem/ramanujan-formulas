"""
Configuration module for Ramanujan-Swarm system.
Contains all constants, settings, and configuration parameters.
"""

import os
from pathlib import Path
from typing import Dict, Any
from mpmath import mp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === PRECISION CONFIGURATION ===
DECIMAL_PRECISION = 1500
mp.dps = DECIMAL_PRECISION

# === SWARM CONFIGURATION ===
SWARM_SIZE = int(os.getenv("SWARM_SIZE", "20"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "500"))
MAX_DISCOVERIES = int(os.getenv("MAX_DISCOVERIES", "5"))

# === MATHEMATICAL CONSTANTS ===
CONSTANTS: Dict[str, Any] = {
    # Fundamental constants
    "pi": mp.pi,
    "e": mp.e,
    "phi": (mp.one + mp.sqrt(5)) / 2,  # Golden ratio

    # Square roots
    "sqrt2": mp.sqrt(2),
    "sqrt3": mp.sqrt(3),
    "sqrt5": mp.sqrt(5),

    # Logarithms
    "ln2": mp.ln(2),
    "ln10": mp.ln(10),

    # Zeta values (Ramanujan used these extensively!)
    "zeta2": mp.zeta(2),  # π²/6
    "zeta3": mp.zeta(3),  # Apéry's constant
    "zeta4": mp.zeta(4),  # π⁴/90
    "zeta5": mp.zeta(5),
    "zeta6": mp.zeta(6),  # π⁶/945

    # Named constants
    "catalan": mp.catalan,  # Catalan's constant
    "euler": mp.euler,  # Euler-Mascheroni constant γ
    "khinchin": mp.khinchin,  # Khinchin's constant
    "glaisher": mp.glaisher,  # Glaisher-Kinkelin constant
    "apery": mp.zeta(3),  # Same as zeta3

    # Special values
    "pisq": mp.pi ** 2,  # π²
    "epi": mp.e ** mp.pi,  # e^π
    "pie": mp.pi ** mp.e,  # π^e
}

# === THRESHOLDS ===
CANDIDATE_THRESHOLD = 1e-12  # Minimum error to keep in gene pool
DISCOVERY_THRESHOLD = 1e-50  # Trigger full verification
GENE_POOL_SIZE = 25  # Number of top candidates to retain

# === NEW: PATTERN SEARCH MODES ===
SEARCH_MODES = {
    "near_integer": True,  # Original: find f(x) ≈ integer
    "series_convergence": True,  # Find fast-converging series
    "sequence_pattern": True,  # Find patterns in f(1), f(2), ...
    "functional_relation": True,  # Find f(x)*g(y) = h(z) patterns
}

# For sequence pattern mode
SEQUENCE_LENGTH = 50  # Evaluate f(n) for n=1..50
MIN_PATTERN_SIGNIFICANCE = 3  # Require pattern in at least 3 consecutive terms

# === SCORING PARAMETERS ===
COMPLEXITY_PENALTY = 0.01  # Reduced penalty to allow for longer theta functions
EXPLORATION_RATE_INITIAL = 0.9  # Increase initial exploration
EXPLORATION_RATE_MIN = 0.6  # Keep exploration high

# === LLM CONFIGURATION ===
SUPPORTED_LLM_MODELS: Dict[str, Dict[str, Any]] = {
    "claude-sonnet-4-5-20250929": {
        "max_tokens": 64000,
        "notes": "Claude Sonnet 4.5 (Sep 2025) – extended context, best coding + agents.",
    },
    "claude-3-5-sonnet-20241022": {
        "max_tokens": 8192,
        "notes": "Claude 3.5 Sonnet (Oct 2024) – stable high-precision reasoning.",
    },
    "claude-3-opus-20240229": {
        "max_tokens": 4096,
        "notes": "Claude 3 Opus (Feb 2024) – legacy premium reasoning tier.",
    },
}
DEFAULT_LLM_MODEL = "claude-sonnet-4-5-20250929"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.9"))
_requested_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "16384"))
_model_cap = SUPPORTED_LLM_MODELS.get(LLM_MODEL, {}).get("max_tokens")
if _model_cap:
    LLM_MAX_TOKENS = min(_requested_max_tokens, _model_cap)
else:
    LLM_MAX_TOKENS = _requested_max_tokens

# === FILE PATHS ===
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
RESULTS_DIR.mkdir(exist_ok=True)

LOG_FILE = RESULTS_DIR / "discoveries.json"
MD_FILE = RESULTS_DIR / "FINAL_REPORT.md"

# === WEB VERIFICATION ===
VERIFICATION_TIMEOUT = int(os.getenv("VERIFICATION_TIMEOUT", "5"))
OEIS_SEARCH_URL = "https://oeis.org/search"
USER_AGENT = "Mozilla/5.0 (Hackathon; Research) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0"

# === VALIDATION ===
def validate_config() -> None:
    """Validate that all required configuration is present."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    if LLM_MODEL not in SUPPORTED_LLM_MODELS:
        supported = ", ".join(SUPPORTED_LLM_MODELS.keys())
        raise ValueError(
            "LLM_MODEL must be one of "
            f"[{supported}]. Please update your .env (current: {LLM_MODEL})."
        )

    if SWARM_SIZE < 1:
        raise ValueError("SWARM_SIZE must be at least 1")
    
    if MAX_ITERATIONS < 1:
        raise ValueError("MAX_ITERATIONS must be at least 1")
    
    if not (0.0 <= LLM_TEMPERATURE <= 1.0):
        raise ValueError(f"LLM_TEMPERATURE must be between 0.0 and 1.0, got {LLM_TEMPERATURE}")

    model_notes = SUPPORTED_LLM_MODELS[LLM_MODEL]["notes"]
    print(f"✅ Configuration validated successfully")
    print(f"   - Model: {LLM_MODEL} ({model_notes})")
    print(f"   - Max Tokens: {LLM_MAX_TOKENS}")
    print(f"   - Swarm Size: {SWARM_SIZE}")
    print(f"   - Max Iterations: {MAX_ITERATIONS}")
    print(f"   - Decimal Precision: {DECIMAL_PRECISION}")
    print(f"   - Results Directory: {RESULTS_DIR.absolute()}")

