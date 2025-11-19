#!/usr/bin/env python3
"""
macOS-specific launcher for Ramanujan-Swarm
Uses python3 explicitly for macOS compatibility
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
        print("   Please install a newer Python version")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")

def install_dependencies():
    """Install dependencies using python3 -m pip"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("\n🔧 Trying with --user flag...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully with --user flag")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ Error installing dependencies with --user: {e2}")
            print("\n💡 Please try manually:")
            print("   python3 -m pip install -r requirements.txt")
            return False

def setup_environment():
    """Setup environment variables"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("\n🔧 Setting up environment...")
    
    # Get Anthropic API key
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
    print("    RAMANUJAN-SWARM (macOS)")
    print("    Mathematical Discovery Engine")
    print("🧬" + "="*50 + "🧬")
    print()
    
    # Check Python
    check_python()
    
    # Check if dependencies are installed
    try:
        import flask
        import flask_socketio
        import mpmath
        print("✅ Dependencies already installed")
    except ImportError:
        print("📦 Installing required dependencies...")
        if not install_dependencies():
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
        print("   3. Try running: python3 -m pip install --upgrade -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
