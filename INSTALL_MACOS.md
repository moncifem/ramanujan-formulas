# 🍎 macOS Installation Guide for Ramanujan-Swarm

## Quick Fix for Your Current Issue

You have `python3` installed but not `python`. Use these commands instead:

```bash
# Navigate to the project directory
cd "/Users/oumark/Downloads/Mobile App/Ramajan"

# Install dependencies using python3
python3 -m pip install -r requirements.txt

# Run setup using python3
python3 setup.py

# Launch the application using python3
python3 run.py
```

## Alternative: Install Python via Homebrew (Recommended)

If you want a more complete Python setup:

### 1. Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python via Homebrew
```bash
brew install python
```

This will give you both `python3` and `pip3` commands, and create symlinks for `python` and `pip`.

### 3. Verify Installation
```bash
python --version
pip --version
```

### 4. Then run the original commands
```bash
cd "/Users/oumark/Downloads/Mobile App/Ramajan"
pip install -r requirements.txt
python setup.py
python run.py
```

## Alternative: Use Python.org Installer

1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or 3.12 for macOS
3. Run the installer
4. This will install both `python3` and `python` commands

## Troubleshooting

### If you get permission errors:
```bash
python3 -m pip install --user -r requirements.txt
```

### If you need to upgrade pip:
```bash
python3 -m pip install --upgrade pip
```

### Check your Python version:
```bash
python3 --version
```

Should be Python 3.8 or higher.

## Quick Start (Using python3)

```bash
# 1. Navigate to project
cd "/Users/oumark/Downloads/Mobile App/Ramajan"

# 2. Install dependencies
python3 -m pip install -r requirements.txt

# 3. Run setup (will ask for Anthropic API key)
python3 setup.py

# 4. Launch application
python3 run.py

# 5. Open browser to http://localhost:5000
```

## What You'll Need

1. **Anthropic API Key**: Get one from https://console.anthropic.com/
2. **Internet Connection**: For OEIS verification (optional)
3. **Modern Web Browser**: Chrome, Firefox, Safari, or Edge

Once running, you'll have a beautiful web interface for mathematical discovery! 🧬
