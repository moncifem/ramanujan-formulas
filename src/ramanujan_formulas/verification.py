"""
Novelty verification module.
Checks whether discovered formulas are novel using online mathematical databases.
"""

import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

import requests
from mpmath import mp

from .config import (
    OEIS_SEARCH_URL,
    USER_AGENT,
    VERIFICATION_TIMEOUT,
    LOG_FILE,
    MD_FILE,
)


class VerificationCache:
    """Cache for verification results to avoid redundant API calls."""
    
    def __init__(self):
        self._cache: Dict[str, dict] = {}
    
    def get(self, key: str) -> Dict[str, Any] | None:
        """Get cached verification result."""
        return self._cache.get(key)
    
    def set(self, key: str, result: dict) -> None:
        """Cache a verification result."""
        self._cache[key] = result
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache


# Global cache instance
_verification_cache = VerificationCache()


def check_novelty_online(value_mpf: mp.mpf) -> dict:
    """
    Check if a mathematical value is novel by querying OEIS database.
    
    Args:
        value_mpf: The mpmath value to check
        
    Returns:
        Dictionary with keys:
        - novel (bool): Whether the value appears to be novel
        - source (str): Data source name
        - oeis (str): "NOVEL", "KNOWN", or "ERROR"
        - error_msg (str, optional): Error message if verification failed
    """
    # Generate signature for caching
    val_str = str(value_mpf).replace('.', '')[:50]
    
    # Check cache first
    if _verification_cache.has(val_str):
        return _verification_cache.get(val_str)
    
    results = {
        "novel": False,
        "source": "Internal",
        "oeis": "UNKNOWN"
    }
    
    headers = {"User-Agent": USER_AGENT}
    
    try:
        # Query first 18 digits (enough for matching)
        query = val_str[:18]
        params = {"q": query, "fmt": "text"}
        
        response = requests.get(
            OEIS_SEARCH_URL,
            params=params,
            headers=headers,
            timeout=VERIFICATION_TIMEOUT
        )
        
        response.raise_for_status()
        
        # Parse response
        if "No sequences found" in response.text or "Results" not in response.text:
            results["oeis"] = "NOVEL"
            results["novel"] = True
        else:
            results["oeis"] = "KNOWN"
            results["novel"] = False
            
    except requests.Timeout:
        results["oeis"] = "ERROR"
        results["error_msg"] = "Request timeout"
    except requests.RequestException as e:
        results["oeis"] = "ERROR"
        results["error_msg"] = str(e)
    except Exception as e:
        results["oeis"] = "ERROR"
        results["error_msg"] = f"Unexpected error: {str(e)}"
    
    # Cache result
    _verification_cache.set(val_str, results)
    
    return results


def save_discovery(discovery: dict) -> None:
    """
    Save a discovery to both JSON log and Markdown report.
    
    Args:
        discovery: Dictionary containing:
            - expression (str): The formula
            - value (str): Computed value
            - error (float): Precision error
            - iteration (int): Discovery iteration
            - verified (bool): Whether it's novel
    """
    # Load existing discoveries
    data = []
    if LOG_FILE.exists():
        try:
            data = json.loads(LOG_FILE.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            print(f"⚠️  Warning: Could not parse {LOG_FILE}, starting fresh")
    
    # Add timestamp
    discovery["timestamp"] = datetime.now().isoformat()
    
    # Append and save JSON
    data.append(discovery)
    LOG_FILE.write_text(json.dumps(data, indent=2), encoding='utf-8')
    
    # Append to Markdown report
    _append_to_markdown_report(discovery, len(data))
    
    print(f"💾 Discovery saved to {LOG_FILE}")


def _append_to_markdown_report(discovery: dict, count: int) -> None:
    """Append a discovery to the markdown report."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Format error in scientific notation
    error = discovery['error']
    if error > 0:
        import math
        try:
            exp = int(math.log10(error))
            error_latex = f"10^{{{exp}}}"
        except (ValueError, OverflowError):
            error_latex = str(error)
    else:
        error_latex = "0"
    
    md_content = f"""
### 🎯 Discovery #{count} - {timestamp}

- **Expression**: `{discovery['expression']}`
- **Error**: ${error_latex}$
- **Value**: `{discovery['value'][:60]}...`
- **Iteration**: {discovery['iteration']}
- **Status**: {'✅ **NOVEL**' if discovery['verified'] else '⚠️ Known in Database'}

---
"""
    
    # Initialize file if needed
    if not MD_FILE.exists():
        MD_FILE.write_text(
            "# 🧬 Mathematical Discovery Report\n\n"
            "## Ramanujan-Swarm Results\n\n",
            encoding='utf-8'
        )
    
    # Append discovery
    with open(MD_FILE, "a", encoding="utf-8") as f:
        f.write(md_content)


def initialize_report() -> None:
    """Initialize the markdown report file if it doesn't exist."""
    if not MD_FILE.exists():
        content = """# 🧬 Mathematical Discovery Report

## Ramanujan-Swarm Results

**Generated**: {datetime}

This report contains mathematical identities discovered by the Ramanujan-Swarm system.
Each entry shows high-precision relationships between fundamental mathematical constants.

---

""".format(datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        MD_FILE.write_text(content, encoding='utf-8')

