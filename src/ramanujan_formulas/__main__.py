"""
Main entry point for running Ramanujan-Swarm.
Can be invoked as: python -m ramanujan_formulas
"""

import asyncio
from .main import main

if __name__ == "__main__":
    asyncio.run(main())

