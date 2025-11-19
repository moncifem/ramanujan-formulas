import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Using Cursor AI instead of external API
    USE_CURSOR_AI = True
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')  # Optional fallback
    FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Ramanujan-Swarm Configuration
    SWARM_SIZE = 20
    MAX_GENERATIONS = 100
    PRECISION_DIGITS = 1500
    ERROR_THRESHOLD_KEEP = 1e-12
    ERROR_THRESHOLD_VERIFY = 1e-50
    ELEGANCE_COMPLEXITY_PENALTY = 0.03
    GENE_POOL_SIZE = 25
    
    # Mathematical constants for discovery
    TARGET_CONSTANTS = {
        'pi': 3.141592653589793238462643383279502884197,
        'e': 2.718281828459045235360287471352662497757,
        'phi': 1.618033988749894848204586834365638117720,
        'gamma': 0.577215664901532860606512090082402431042,
        'zeta3': 1.202056903159594285399738161511449990765
    }
