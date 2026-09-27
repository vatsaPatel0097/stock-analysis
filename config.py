"""Project thresholds and placeholder model ids. No network I/O."""

STOP_PCT = 0.02
WINDOW_DAYS = 15
MIN_TARGET_PROB = 0.35
LLM_DAILY_CAP = 950

# R2 gates. ATR limits are percentage points (2.0 means 2% of price).
# Stop, chase, and target moves in analytics are fractions (0.02 means 2%).
ATR_PCT_MAX = 2.0
ATR_PCT_MIN = 0.4
MIN_AVG_TRADED_VALUE = 50_000_000  # ₹5 crore
MIN_SAMPLE_SIZE = 50
CHASE_PCT = 0.015

# Free OpenRouter models (R3.5). Recorded in DECISIONS.md on 2026-09-27.
# Ultra is the strongest free reasoning model that still returns strict JSON.
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_FALLBACK_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_REASONING_EFFORT = "high"
