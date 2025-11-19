# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ramanujan-Swarm is an agentic AI system for discovering novel mathematical identities through evolutionary search. It uses LangGraph's parallel orchestration with a genetic algorithm to find high-precision relationships between fundamental constants (π, e, ϕ, ζ(3), etc.).

## Development Commands

The project uses Python 3.10+ and is managed with `uv` (or standard Python tooling).

```bash
# Install dependencies
pip install -r requirements.txt
# or
pip install -e .

# Run the main application
python main.py

# Test basic functionality
python test_basic.py

# Test provider support
python test_providers.py
```

## LLM Provider Configuration

The system supports **Claude (Anthropic)**, **Gemini (Google)**, **AWS Bedrock**, and **Blackbox AI** as LLM providers:

**Using Claude (Direct API):**
```env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Using Gemini:**
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key-here
```

**Using Blackbox AI (Recommended - Easiest Setup):**
```env
LLM_PROVIDER=blackbox
BLACKBOX_API_KEY=your-blackbox-key-here
```

**Using AWS Bedrock:**
```env
LLM_PROVIDER=bedrock
AWS_BEARER_TOKEN_BEDROCK=your-bedrock-token-here
AWS_REGION=eu-west-3  # Or your region
```

### Provider Comparison

| Provider | Setup Difficulty | Models Available | Notes |
|----------|-----------------|------------------|-------|
| **Blackbox AI** | ✅ Easy | Claude 3.5/3.7, GPT-4, etc. | No payment method required, unified API |
| Claude Direct | ✅ Easy | Claude 3.5 Sonnet | Requires Anthropic credits |
| Gemini | ✅ Easy | Gemini 2.0 Flash | May have quota limits |
| AWS Bedrock | ⚠️ Complex | Claude Sonnet 4.5 | Requires AWS payment setup, inference profiles |

**Blackbox AI Benefits:**
- Access to Claude models without AWS complexity
- OpenAI-compatible API (easy integration)
- No payment instrument setup required
- Get API key at: https://www.blackbox.ai/dashboard/docs

**Important Bedrock Notes:**
- Claude Sonnet 4.5 uses **inference profiles**: `eu.anthropic.claude-sonnet-4-5-20250929-v1:0` (EU), `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (US)
- Requires valid payment method in AWS account
- Temperature must be ≤ 1.0 (config automatically set to 1.0 for Bedrock)
- May require AWS Marketplace subscription for Claude models

See `DUAL_PROVIDER_SUPPORT.md` for detailed information on choosing and configuring providers.

## Core Architecture

### Map-Reduce + Evolutionary Pattern

The system is designed as a **parallel genetic swarm** using LangGraph v0.2's Send API:

1. **Map Phase**: 20+ parallel agents (Claude 3.5 Sonnet, temp 1.1) explore the search space
   - **Explorers**: Generate novel symbolic expressions
   - **Mutators**: Modify expressions from the gene pool
   - **Hybrids**: Combine both strategies

2. **Reduce Phase**: Central validator node performs precision filtering
   - Syntax hash deduplication (structural equivalence)
   - High-precision numeric evaluation (mpmath at 1500 decimal places)
   - Elegance scoring: `Score = Error × (1 + 0.03 × Length)`

3. **Memory System**:
   - Gene pool retains top 25 candidates per generation
   - LangGraph MemorySaver checkpoints state across generations
   - OEIS verification for discovered identities

### Dual-Threshold Strategy

- **10⁻¹²**: Keep "interesting" approximations for evolution
- **10⁻⁵⁰**: Trigger full verification and logging to `FINAL_REPORT.md`

## Technology Stack

- **Orchestration**: LangGraph v0.2.45+ with MemorySaver
- **LLM**: LangChain + Anthropic Claude 3.5 Sonnet (Temperature 1.1)
- **Computation**: mpmath (1500+ digit precision)
- **Verification**: requests + BeautifulSoup for OEIS/ISC checking
- **Output**: Continuous markdown report generation

## Implementation Status

The project is in early stages - the README describes the target architecture but main.py only contains a placeholder. When implementing:

- The swarm dispatch logic should use LangGraph's Send API for parallel agent spawning
- The validator node is the "central CPU" that aggregates results from all parallel agents
- The gene pool is shared memory accessible across all generations
- Performance target: 2000+ expressions/min across parallel threads
