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

# Placeholders until Phase 2 model pick (R3.5). Record the real ids in DECISIONS.md.
OPENROUTER_MODEL = "CHANGE_ME/free-model"
OPENROUTER_FALLBACK_MODEL = "CHANGE_ME/free-fallback"
