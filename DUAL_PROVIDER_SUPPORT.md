# Dual LLM Provider Support

## Overview

The Ramanujan-Swarm system now supports **both Claude (Anthropic) and Gemini (Google)** as LLM providers. You can easily switch between them using environment variables.

## ✅ What's New

### Supported Providers

1. **Claude (Anthropic)** - `LLM_PROVIDER=claude`
   - Model: `claude-3-5-sonnet-20240620` (default)
   - Excellent mathematical reasoning
   - Creative expression generation
   - Premium pricing

2. **Gemini (Google)** - `LLM_PROVIDER=gemini`
   - Model: `gemini-2.0-flash-exp` (default)
   - Fast inference
   - Cost-effective
   - Strong mathematical capabilities

## 🚀 How to Use

### Option 1: Using Claude

Create or edit `.env`:

```env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Option 2: Using Gemini

Create or edit `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key-here
```

### Custom Model Selection

You can override the default model:

```env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-your-key-here
LLM_MODEL=claude-3-opus-20240229
```

Or for Gemini:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key-here
LLM_MODEL=gemini-1.5-pro
```

## 🔧 Technical Implementation

### Architecture

The system uses a factory pattern in `BaseAgent._initialize_llm()`:

```python
def _initialize_llm(self):
    if config.llm_provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            api_key=config.anthropic_api_key,
        )
    elif config.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            google_api_key=config.gemini_api_key,
        )
```

### API Key Validation

The system validates API keys on startup:

```python
config.validate_api_keys()
```

This ensures you have the correct key set before running.

## 📊 Performance Comparison

### Claude 3.5 Sonnet

**Pros:**
- ✅ Superior mathematical creativity
- ✅ Better at complex prompt following
- ✅ Excellent at generating novel patterns

**Cons:**
- 💰 Higher cost (~$3/million input tokens, ~$15/million output)
- ⏱️ Slightly slower inference

**Best for:**
- Research-grade discoveries
- Novel identity exploration
- When quality > cost

### Gemini 2.0 Flash

**Pros:**
- ✅ Very fast inference
- ✅ Cost-effective (~$0.075/million input tokens, ~$0.30/million output)
- ✅ Good mathematical reasoning

**Cons:**
- ⚠️ May be less creative than Claude
- ⚠️ Prompt adherence can vary

**Best for:**
- Large-scale experiments
- Cost-sensitive projects
- Rapid iteration

## 🧪 Testing

All provider tests pass:

```bash
python test_providers.py
```

Tests verify:
- ✅ Configuration switching
- ✅ Agent initialization with both providers
- ✅ API key validation
- ✅ Custom model overrides

## 📈 Cost Estimates

### Default Configuration (10 agents, 20 generations)

**With Claude 3.5 Sonnet:**
- ~600 API calls
- ~300K input tokens + ~60K output tokens
- **Estimated cost: $1.80**

**With Gemini 2.0 Flash:**
- ~600 API calls
- ~300K input tokens + ~60K output tokens
- **Estimated cost: $0.04**

💡 **Gemini is ~45x cheaper for the same workload!**

## 🔄 Switching Providers Mid-Project

You can run experiments with different providers:

```bash
# First run with Gemini (fast exploration)
LLM_PROVIDER=gemini python main.py

# Then refine with Claude (quality refinement)
LLM_PROVIDER=claude python main.py
```

The gene pool persists across runs via checkpointing!

## 🎯 Recommended Workflow

### 1. Initial Exploration (Gemini)
```env
LLM_PROVIDER=gemini
SWARM_SIZE=20
MAX_GENERATIONS=50
```

- Fast iteration
- Low cost
- Broad search space coverage

### 2. Refinement (Claude)
```env
LLM_PROVIDER=claude
SWARM_SIZE=10
MAX_GENERATIONS=20
```

- Deep mathematical reasoning
- Creative mutations
- Quality over quantity

### 3. Production Discovery (Hybrid)

Alternate between providers or run both in parallel for complementary strengths.

## 📚 Dependencies

Both providers are included in `requirements.txt`:

```txt
langchain-anthropic>=0.3.0  # For Claude
langchain-google-genai>=2.0.0  # For Gemini
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

## 🔐 API Keys

### Getting API Keys

**Claude (Anthropic):**
1. Visit: https://console.anthropic.com/
2. Create account
3. Generate API key
4. Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`

**Gemini (Google):**
1. Visit: https://makersuite.google.com/app/apikey
2. Create API key
3. Add to `.env`: `GEMINI_API_KEY=...`

## ⚠️ Important Notes

### Rate Limits

- **Claude**: 5,000 requests/min (tier 1), 10,000 requests/min (tier 2+)
- **Gemini**: 60 requests/min (free), 1,000+ (paid)

Adjust `SWARM_SIZE` accordingly to avoid rate limiting.

### Model Availability

- **Claude**: Always available (with API key)
- **Gemini**: `gemini-2.0-flash-exp` is experimental, may change

### Context Windows

- **Claude 3.5 Sonnet**: 200K tokens
- **Gemini 2.0 Flash**: 1M tokens

Both are more than sufficient for our prompts (~1K tokens each).

## 🐛 Troubleshooting

### Error: ANTHROPIC_API_KEY must be set

```bash
# Solution: Check your .env file
cat .env | grep ANTHROPIC_API_KEY
```

### Error: GEMINI_API_KEY must be set

```bash
# Solution: Check your .env file
cat .env | grep GEMINI_API_KEY
```

### Error: Unknown LLM provider

```bash
# Solution: Ensure LLM_PROVIDER is "claude" or "gemini"
cat .env | grep LLM_PROVIDER
```

### Module not found: langchain_google_genai

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

## 📖 Examples

### Example 1: Quick Test with Gemini

```bash
# .env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key
SWARM_SIZE=5
MAX_GENERATIONS=5
```

```bash
python main.py
```

Expected: Fast execution, ~$0.01 cost

### Example 2: Research Run with Claude

```bash
# .env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=your-key
SWARM_SIZE=20
MAX_GENERATIONS=50
```

```bash
python main.py
```

Expected: High-quality results, ~$5-10 cost

### Example 3: Hybrid Approach

```bash
# Phase 1: Gemini exploration
echo "LLM_PROVIDER=gemini" > .env
python main.py

# Phase 2: Claude refinement
echo "LLM_PROVIDER=claude" > .env
python main.py
```

## 🎉 Summary

✅ **Dual provider support implemented**
✅ **Easy switching via environment variables**
✅ **Both providers tested and working**
✅ **Cost-effective option (Gemini) available**
✅ **High-quality option (Claude) available**
✅ **Flexible workflow options**

Choose the provider that best fits your needs:
- **Gemini** for cost-effective exploration
- **Claude** for premium quality discoveries
- **Both** for hybrid workflows

---

Generated with Claude Code (claude.ai/code)
