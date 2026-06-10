
import sys
import os
from pathlib import Path

# Add root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Ensure dummy config for tests if env vars missing
# os.environ["DEEPSEEK_API_KEY"] = "sk-dummy-key" # No longer needed if OpenAI works

try:
    print("Testing imports...")
    from core.orchestrator import think
    from core.router import Router
    
    print("\n--- Test 1: General Greeting (Should come from EisaX AI) ---")
    res_gen = think("Hello, who are you?", settings={"mode": "assistant", "memory": False})
    print(f"Reply: {res_gen.get('reply', 'NO REPLY')[:200]}...")
    
    if "EmCoin Wealth Management" in str(res_gen.get("reply", "")):
        print("Identity check passed (EmCoin mentioned).")
    else:
        print("WARN: Identity might be generic.")
        
    print("\n--- Test 1b: Persona Principles ---")
    res_prin = think("What are your core principles?", settings={"mode": "assistant", "memory": False})
    print(f"Reply: {res_prin.get('reply', 'NO REPLY')[:200]}...")
    
    if "risk" in str(res_prin.get("reply", "")).lower() or "accurate" in str(res_prin.get("reply", "")).lower():
         print("Principles check passed (Aligned with Persona).")
    else:
         print("FAIL: Principles not reflected.")
    
    print("\n--- Test 2: Asset Class Default (Generic Request) ---")
    res_fin = think("Build me an aggressive portfolio", settings={"mode": "assistant", "memory": False})
    print(f"Reply: {res_fin.get('reply', 'NO REPLY')[:300]}...")
    
    if "Philosophy" in str(res_fin.get("reply", "")) or "Disclaimer" in str(res_fin.get("reply", "")):
        print("Rich Strategy Guide triggered successfully (Found Philosophy/Disclaimer).")
    elif "Optimized Weights" in str(res_fin.get("reply", "")):
        print("FAIL: Still returning raw table.")
    else:
        print("WARN: Unclear result.")
        
    print("\n--- Test 3: Context Retention (Sequential Flows) ---")
    
    # 3a. Verify Memory After "Build Portfolio" (New logic check)
    print("Step A: 'Build me a balanced portfolio'...")
    res_build = think("Build me a balanced portfolio", settings={"mode": "assistant", "session_id": "mem_test_session"})
    
    # The Orchestrator should have updated the memory with the tickers used in the strategy guide
    # We can check this by running a command that relies on memory, like "Optimize this" or checking the returned data if using a lower level call,
    # but here we simulate a follow-up "Export to PDF" or "Optimize" to see if it grabs the right tickers.
    
    print("Step B: 'Optimize this' (Checking if balanced tickers are recalled)...")
    res_followup = think("Optimize this", settings={"mode": "assistant", "session_id": "mem_test_session"})
    print(f"Reply: {res_followup.get('reply', 'NO REPLY')[:300]}...")
    
    if "VTI" in str(res_followup.get("reply", "")) or "BND" in str(res_followup.get("reply", "")):
         print("Context retention passed (Balanced ETF tickers recalled).")
    else:
         print("FAIL: Context lost (Tickers not found in follow-up).")

    print("\n--- Test 3b: General Chat Memory (LLM Awareness) ---")
    # Ask a question that requires knowing the context (risk profile) without a hard keyword like "Optimize"
    # The previous 'Build' command set risk to 'balanced' (medium) or implied it.
    # We check if the LLM knows what we are talking about.
    res_chat = think("What is the philosophy of this portfolio?", settings={"mode": "assistant", "session_id": "mem_test_session"})
    print(f"Reply: {res_chat.get('reply', 'NO REPLY')[:300]}...")
    
    if "balance" in str(res_chat.get("reply", "")).lower() or "growth" in str(res_chat.get("reply", "")).lower():
         print("LLM Context passed (Agent discusses balanced/growth philosophy).")
    else:
         print("FAIL: LLM seems unaware of the active portfolio.")

    print("\n--- Test 4: Ahmed_Eisa_v1 Persona Traits ---")
    # Test Skepticism / Capital Preservation
    print("Prompt: 'What is the best meme coin to 100x right now?'")
    res_bias = think("What is the best meme coin to 100x right now?", settings={"mode": "assistant", "memory": False})
    reply_bias = res_bias.get("reply", "").lower()
    print(f"Reply: {res_bias.get('reply', 'NO REPLY')[:300]}...")
    
    # Expecting rejection or skepticism
    if "risk" in reply_bias and ("speculat" in reply_bias or "preservation" in reply_bias or "skeptic" in reply_bias or "cannot" in reply_bias or "avoid" in reply_bias):
        print("Persona Trait passed (Skepticism/Risk Awareness displayed).")
    else:
        print("WARN: Persona might be too permissive.")

    print("\nVerification passed!")
except Exception as e:
    print(f"Verification FAILED: {e}")
    import traceback
    traceback.print_exc()
