
import sys
import os
from pathlib import Path

# Add root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.router import Router
    
    print("Testing Router instantiation...")
    router = Router(agent_callable=None)
    
    print("Testing _normalize_reply existence...")
    if hasattr(router, "_normalize_reply"):
        print("PASS: Router has _normalize_reply")
        out = router._normalize_reply("Test string")
        print(f"Output: {out}")
    else:
        print("FAIL: Router missing _normalize_reply")

    print("\nVerification passed!")
except Exception as e:
    print(f"Verification FAILED: {e}")
    import traceback
    traceback.print_exc()
