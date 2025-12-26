# helpers.py

import re
from langdetect import detect, LangDetectException

# -----------------------------------
# HEADERS FOR REQUESTS
# -----------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}

# -----------------------------------
# CLEAN TEXT
# -----------------------------------
def clean_text(text):
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

# -----------------------------------
# FILTER ENGLISH ONLY
# -----------------------------------
def is_english(text):
    """Return True only if detected language is English and text is ASCII-like."""
    t = (text or "").strip()
    if len(t) < 4:
        return False

    # reject non-ASCII (Hindi, Chinese, emojis, etc.)
    if re.search(r'[^\x00-\x7F]', t):
        return False

    try:
        return detect(t) == "en"
    except LangDetectException:
        return False
