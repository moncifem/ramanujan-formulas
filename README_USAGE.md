# Ramanujan-Swarm Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Set Up Environment

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and configure your LLM provider:

**Option A: Using Claude (Anthropic)**
```env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-your-key-here
SWARM_SIZE=10
MAX_GENERATIONS=20
GENE_POOL_SIZE=15
TARGET_CONSTANTS=pi,e,phi
PRECISION_DPS=100
```

**Option B: Using Gemini (Google)**
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key-here
SWARM_SIZE=10
MAX_GENERATIONS=20
GENE_POOL_SIZE=15
TARGET_CONSTANTS=pi,e,phi
PRECISION_DPS=100
```

The system will automatically use the provider you specify in `LLM_PROVIDER`.

### 3. Run the Swarm

```bash
python main.py
```

## What Happens

The system will:

1. **Initialize** a swarm of 10 agents (configurable)
2. **Generation Loop**:
   - **Dispatch Phase**: Fan out to N parallel agents using LangGraph Send API
   - **Agent Phase**: Each agent (Explorer/Mutator/Hybrid) uses Claude to generate mathematical expressions
   - **Validation Phase**: Parse, evaluate at high precision (100 decimals), deduplicate
   - **Scoring Phase**: Apply elegance scoring (error × complexity penalty)
   - **Gene Pool Update**: Keep top 15 candidates across all generations
3. **Discovery Check**: If any expression has error < 1e-50, mark as discovery
4. **Continue**: Loop for up to 20 generations

## Output

Results are saved to:

- `outputs/FINAL_REPORT.md` - Markdown report with discoveries and top candidates

## Key Innovations

### 1. Parallel Genetic Swarm with Send API

The core innovation is the **Map-Reduce pattern** using LangGraph's Send API:

```python
# In dispatch node
sends = [Send("agent_worker", {...}) for i in range(swarm_size)]
return sends
```

This creates N parallel agent instances that:
- Run completely independently
- Each calls Claude to generate expressions
- All results automatically aggregate via `operator.add` reducer

### 2. Three Agent Types

- **Explorer (40%)**: Generates completely novel expressions
- **Mutator (40%)**: Refines expressions from gene pool
- **Hybrid (20%)**: Combines both strategies

### 3. Dual-Threshold System

- **1e-12**: Keep for evolution (gene pool)
- **1e-50**: Mark as significant discovery

### 4. High-Precision Mathematics

Uses `mpmath` for 100+ decimal place accuracy, ensuring discoveries are genuine.

### 5. Elegance Scoring

```
Score = Error × (1 + 0.03 × Length)
```

Penalizes unnecessarily complex expressions, driving toward Ramanujan-style elegance.

## Configuration Options

Edit `.env` to customize:

- `LLM_PROVIDER`: Choose "claude" or "gemini" (default: claude)
- `ANTHROPIC_API_KEY`: Your Claude API key (required if using Claude)
- `GEMINI_API_KEY`: Your Gemini API key (required if using Gemini)
- `LLM_MODEL`: Override default model (optional)
  - Claude default: `claude-3-5-sonnet-20240620`
  - Gemini default: `gemini-2.0-flash-exp`
- `SWARM_SIZE`: Number of parallel agents (default: 10)
- `MAX_GENERATIONS`: Generation limit (default: 20)
- `GENE_POOL_SIZE`: Top N candidates to keep (default: 15)
- `TARGET_CONSTANTS`: Comma-separated constants (pi,e,phi,euler,apery,catalan)
- `PRECISION_DPS`: Decimal precision (default: 100)

### Choosing Between Claude and Gemini

**Claude (Anthropic)**
- ✅ Excellent mathematical reasoning
- ✅ Creative expression generation
- ✅ Good at following complex prompts
- 💰 More expensive per token

**Gemini (Google)**
- ✅ Fast inference
- ✅ Good mathematical capabilities
- ✅ More cost-effective
- ✅ Latest version: gemini-2.0-flash-exp

## Performance

Expected performance with default settings:
- **10 agents × 3 expressions × 20 generations = 600 total expressions**
- **Runtime**: ~5-10 minutes (depends on API latency)
- **Cost**: ~$0.50-1.00 (based on Claude 3.5 Sonnet pricing)

## Monitoring Progress

The system prints:
- Generation number
- Number of proposals received
- Number of valid/deduplicated expressions
- Best error for each generation
- Top candidate preview

## Troubleshooting

### API Key Error

Make sure your `.env` file contains a valid Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Import Errors

Make sure all dependencies are installed:

```bash
uv sync
# or
pip install -e .
```

### ModuleNotFoundError

Make sure you're running from the project root directory.

## Next Steps

Once you have successful runs:

1. **Increase swarm size** for more parallel exploration
2. **Increase generations** for deeper evolution
3. **Increase precision** (e.g., 500 or 1500 decimals)
4. **Add more constants** (apery, euler, catalan)
5. **Implement OEIS verification** to check against known sequences

## Architecture Diagram

```
┌─────────────┐
│ Initialize  │
└──────┬──────┘
       │
┌──────▼──────────┐
│    Dispatch     │  ← Creates Send() messages
└────────┬────────┘
         │
    ┌────┴─────┐ Fan-out (MAP phase)
    │          │
┌───▼──┐  ┌───▼──┐  ... (N agents in parallel)
│Agent1│  │Agent2│
│ (E)  │  │ (M)  │
└───┬──┘  └───┬──┘
    │          │
    └────┬─────┘ Aggregate (REDUCE phase)
         │
┌────────▼────────┐
│   Validator     │ ← Dedupe, score, filter
└────────┬────────┘
         │
┌────────▼────────┐
│ Gene Pool Update│ ← Keep top N
└────────┬────────┘
         │
┌────────▼────────┐
│Discovery Check? │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
  Save    Continue → Loop back to Dispatch
```

## Contributing

This is a research prototype. Contributions welcome:

1. OEIS verification integration
2. More sophisticated mutation strategies
3. Visualization of gene pool evolution
4. Support for custom mathematical functions
5. Distributed execution across multiple machines

## License

See LICENSE file.
