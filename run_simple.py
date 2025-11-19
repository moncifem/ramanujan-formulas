#!/usr/bin/env python3
"""
Simplified launcher for Ramanujan-Swarm (no LangGraph dependencies)
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")

def upgrade_pip():
    """Upgrade pip to latest version"""
    print("\n🔧 Upgrading pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ Pip upgraded successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not upgrade pip: {e}")

def install_dependencies():
    """Install dependencies using simplified requirements"""
    print("\n📦 Installing dependencies (simplified version)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_simple.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("\n🔧 Trying with --user flag...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-r", "requirements_simple.txt"])
            print("✅ Dependencies installed successfully with --user flag")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ Error installing dependencies with --user: {e2}")
            return False

def setup_environment():
    """Setup environment variables"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("\n🔧 Setting up environment...")
    print("\n🔑 You need an Anthropic API key to use Claude 3.5 Sonnet")
    print("   Get one at: https://console.anthropic.com/")
    api_key = input("   Enter your Anthropic API key: ").strip()
    
    if not api_key:
        print("❌ API key is required for the system to work")
        return False
    
    # Create .env file
    import secrets
    secret_key = secrets.token_urlsafe(32)
    
    env_content = f"""ANTHROPIC_API_KEY={api_key}
FLASK_SECRET_KEY={secret_key}
DEBUG=True
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ Environment file created")
    return True

def main():
    """Main launcher function"""
    print("🧬" + "="*50 + "🧬")
    print("    RAMANUJAN-SWARM (Simplified)")
    print("    Mathematical Discovery Engine")
    print("🧬" + "="*50 + "🧬")
    print()
    
    # Check Python
    check_python()
    
    # Upgrade pip first
    upgrade_pip()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please try:")
        print("   python3 -m pip install --upgrade pip")
        print("   python3 -m pip install -r requirements_simple.txt")
        sys.exit(1)
    
    # Setup environment if needed
    if not setup_environment():
        sys.exit(1)
    
    # Check configuration
    try:
        sys.path.insert(0, '.')
        from config import Config
        if not Config.ANTHROPIC_API_KEY:
            print("❌ API key not configured properly")
            sys.exit(1)
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    print("\n🚀 Starting Ramanujan-Swarm...")
    print("🌐 Web interface will be available at: http://localhost:5000")
    print("🔬 Ready for mathematical discovery!")
    print("\n💡 Press Ctrl+C to stop the server")
    print()
    
    # Launch the app
    try:
        from app import app, socketio, Config
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=Config.DEBUG,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        print("\n👋 Ramanujan-Swarm stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure your API key is correct")
        print("   2. Check that port 5000 is available")
        print("   3. Try running: python3 -m pip install --upgrade -r requirements_simple.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
