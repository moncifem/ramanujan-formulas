#!/bin/bash
# Quick fix for .env file to use correct Claude model

echo "Fixing .env file..."

# Backup existing .env
if [ -f .env ]; then
    cp .env .env.backup
    echo "Backed up .env to .env.backup"
fi

# Replace old model names with the correct one
if [ -f .env ]; then
    # Use sed to replace in-place (works on both Mac and Linux)
    sed -i.tmp 's/claude-3-5-sonnet-20241022/claude-sonnet-4-5-20250929/g' .env
    sed -i.tmp 's/claude-4-5-sonnet-20250929/claude-sonnet-4-5-20250929/g' .env
    sed -i.tmp 's/LLM_MAX_TOKENS=4096/LLM_MAX_TOKENS=16384/g' .env
    rm -f .env.tmp
else
    echo "No .env file found. Creating from template..."
    cp env.example .env
fi

echo ""
echo "✅ Fixed! Your .env now uses: claude-sonnet-4-5-20250929"
echo ""
echo "Current settings:"
grep -E "LLM_MODEL=|LLM_MAX_TOKENS=" .env

echo ""
echo "Run: uv run main.py"
