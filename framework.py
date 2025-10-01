"""Legacy framework file - functionality moved to src/core/.

This file is kept for backward compatibility. 
Please use the new modular structure in src/ directory.

Author: Niels Vyncke
"""

import sys
import os

# Add src to Python path for backward compatibility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import everything from the new core module
from core import *

print("Warning: Using legacy framework.py. Please update imports to use 'from src.core import ...'")

# All functionality has been moved to src/core/
# This file exists only for backward compatibility
