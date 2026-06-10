import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# General AI Configuration
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================================================
# OpenRouter Configuration
# ============================================================
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ============================================================
# DeepSeek API Configuration
# ============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# Main model — V4-flash (fast, ~1-3s) for standard analytics path.
# V4-pro is reserved for deep institutional report generation only.
DEFAULT_MODEL = os.getenv("MODEL_NAME", "deepseek-v4-flash")

# Specific model constants
DEEPSEEK_MODEL_REASONING = "deepseek-v4-pro"    # deep reasoning — slow, for institutional reports
DEEPSEEK_MODEL_CHAT      = "deepseek-v4-flash"  # standard chat / analytics — fast path
DEEPSEEK_MODEL_FAST      = "deepseek-v4-flash"  # explicit fast alias

# ============================================================
# Portfolio Defaults
# ============================================================
DEFAULT_START = os.getenv("DEFAULT_START", "2022-01-01")
DEFAULT_MAX_W = float(os.getenv("DEFAULT_MAX_W", "0.20"))
DEFAULT_MIN_W = float(os.getenv("DEFAULT_MIN_W", "0.0"))
DEFAULT_MIN_ASSETS = int(float(os.getenv("DEFAULT_MIN_ASSETS", "4")))
DEFAULT_SEED_W = float(os.getenv("DEFAULT_SEED_W", "0.02"))
DEFAULT_RF = float(os.getenv("DEFAULT_RF", "0.0"))

# ============================================================
# Thinking Pipeline Settings
# ============================================================
THINKING_TIMEOUT_S = 45

MAX_TOKENS_REASON = 800
MAX_TOKENS_CHALLENGE = 600
MAX_TOKENS_STRUCTURE = 500

TEMP_REASON = 0.3
TEMP_CHALLENGE = 0.4
TEMP_STRUCTURE = 0.2

# ============================================================
# Validation
# ============================================================
if not DEEPSEEK_API_KEY:
    print("[WARNING] DEEPSEEK_API_KEY is missing from environment variables.")
