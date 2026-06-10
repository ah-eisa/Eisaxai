import sys
import os
from unittest.mock import MagicMock

# MOCK MISSING DEPS
sys.modules["pypfopt"] = MagicMock()
sys.modules["pypfopt.expected_returns"] = MagicMock()
sys.modules["pypfopt.risk_models"] = MagicMock()
sys.modules["pypfopt.efficient_frontier"] = MagicMock()
sys.modules["empyrical"] = MagicMock()
sys.modules["cvxpy"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    print("Attempting to import FinancialAgent...")
    from core.agents.finance import FinancialAgent
    print("SUCCESS: FinancialAgent imported.")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
