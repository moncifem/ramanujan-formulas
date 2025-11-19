#!/usr/bin/env python3
"""
Quick launcher for Ramanujan-Swarm
"""

import os
import sys
from pathlib import Path

def main():
    """Launch the Ramanujan-Swarm application"""
    
    # Check if .env exists
    if not Path('.env').exists():
        print("🔧 No .env file found. Running setup...")
        os.system('python setup.py')
        print()
    
    # Check if API key is configured
    try:
        from config import Config
        if not Config.ANTHROPIC_API_KEY:
            print("❌ ANTHROPIC_API_KEY not configured")
            print("   Please run: python setup.py")
            sys.exit(1)
    except ImportError:
        print("❌ Configuration error. Please run: python setup.py")
        sys.exit(1)
    
    print("🧬 Starting Ramanujan-Swarm...")
    print("🌐 Web interface will be available at: http://localhost:5000")
    print("🔬 Ready for mathematical discovery!")
    print()
    
    # Launch the app
    try:
        from app import app, socketio
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
        sys.exit(1)

if __name__ == "__main__":
    main()
