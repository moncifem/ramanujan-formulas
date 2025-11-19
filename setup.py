#!/usr/bin/env python3
"""
Setup script for Ramanujan-Swarm
Automated setup and configuration
"""

import os
import sys
import subprocess
import secrets
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🧬" + "="*60 + "🧬")
    print("    RAMANUJAN-SWARM SETUP")
    print("    Autonomous Mathematical Discovery via Genetic Agents")
    print("🧬" + "="*60 + "🧬")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def setup_environment():
    """Setup environment variables"""
    print("\n🔧 Setting up environment...")
    
    env_file = Path(".env")
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        response = input("   Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("   Skipping environment setup")
            return
    
    # Get Anthropic API key
    api_key = input("\n🔑 Enter your Anthropic API key: ").strip()
    if not api_key:
        print("❌ API key is required for the system to work")
        sys.exit(1)
    
    # Generate secret key
    secret_key = secrets.token_urlsafe(32)
    
    # Create .env file
    env_content = f"""ANTHROPIC_API_KEY={api_key}
FLASK_SECRET_KEY={secret_key}
DEBUG=True
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ Environment file created")

def verify_setup():
    """Verify the setup is working"""
    print("\n🔍 Verifying setup...")
    
    try:
        # Test imports
        import flask
        import flask_socketio
        import langchain_anthropic
        import mpmath
        import requests
        import beautifulsoup4
        print("✅ All required packages imported successfully")
        
        # Test environment
        from config import Config
        if Config.ANTHROPIC_API_KEY:
            print("✅ API key configured")
        else:
            print("❌ API key not found in configuration")
            return False
            
        print("✅ Configuration loaded successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n🚀 Setup Complete!")
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS:")
    print("="*60)
    print("\n1. Start the application:")
    print("   python app.py")
    print("\n2. Open your web browser:")
    print("   http://localhost:5000")
    print("\n3. Select a target constant (π, e, φ, γ, ζ(3))")
    print("\n4. Click 'Start Discovery' to launch the swarm!")
    print("\n5. Watch real-time mathematical discoveries")
    print("\n6. Export results when satisfied")
    print("\n" + "="*60)
    print("FEATURES:")
    print("="*60)
    print("• 20+ Parallel AI Agents")
    print("• Evolutionary Genetic Algorithm") 
    print("• High-Precision Mathematics (1500+ digits)")
    print("• Real-time Web Interface")
    print("• OEIS Verification")
    print("• Discovery Export")
    print("\n" + "="*60)
    print("TROUBLESHOOTING:")
    print("="*60)
    print("• Check .env file for correct API key")
    print("• Ensure port 5000 is available")
    print("• See README.md for detailed documentation")
    print("• Enable DEBUG=True for detailed logging")
    print("\n🧬 Happy Mathematical Discovery! 🧬")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Setup environment
    setup_environment()
    
    # Verify setup
    if verify_setup():
        print_usage_instructions()
    else:
        print("\n❌ Setup verification failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
