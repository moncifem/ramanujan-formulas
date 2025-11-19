#!/usr/bin/env python3
"""
Quick script to update your .env file to use the correct Claude model.
"""

import os
from pathlib import Path

def update_env_file():
    env_file = Path(".env")
    
    if not env_file.exists():
        print("❌ No .env file found. Creating from template...")
        template = Path("env.example")
        if template.exists():
            env_file.write_text(template.read_text())
        return
    
    # Read current content
    content = env_file.read_text()
    
    # Replace old model names with the new one
    replacements = [
        ("claude-3-5-sonnet-20241022", "claude-sonnet-4-5-20250929"),
        ("claude-4-5-sonnet-20250929", "claude-sonnet-4-5-20250929"),  # Also fix wrong format
        ("LLM_MAX_TOKENS=4096", "LLM_MAX_TOKENS=16384"),  # Update token limit
    ]
    
    original_content = content
    for old, new in replacements:
        content = content.replace(old, new)
    
    if content != original_content:
        # Backup old file
        backup = Path(".env.backup")
        backup.write_text(original_content)
        print(f"📋 Backed up old .env to {backup}")
        
        # Write updated content
        env_file.write_text(content)
        print("✅ Updated .env file with correct model: claude-sonnet-4-5-20250929")
        print("✅ Updated max tokens to: 16384")
    else:
        print("ℹ️  .env file already has correct settings")
    
    # Show current settings
    print("\n📝 Current .env settings:")
    for line in content.split('\n'):
        if 'LLM_MODEL=' in line or 'LLM_MAX_TOKENS=' in line:
            print(f"   {line}")

if __name__ == "__main__":
    update_env_file()
    print("\n🚀 Now run: uv run main.py")
