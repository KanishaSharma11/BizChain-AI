import requests
import random
import feedparser
from .helpers import clean_text, is_english

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)",
    "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)"
]

def get_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

# ---------------------------------------------------------
# SUPER-STABLE TWITTER SCRAPER USING NITTER RSS FEEDS
# ---------------------------------------------------------
def scrape_twitter(query, limit=120, debug=False):

    def log(*m):
        if debug:
            print("[TWITTER RSS]", *m)

    nitter_instances = [
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://nitter.cz"
    ]

    results = []
    seen = set()
    q = query.replace(" ", "+")

    for base in nitter_instances:
        try:
            url = f"{base}/search/rss?f=tweets&q={q}"
            log("Fetching:", url)

            r = requests.get(url, headers=get_headers(), timeout=10)
            if r.status_code != 200:
                continue

            feed = feedparser.parse(r.text)
            log("Items found:", len(feed.entries))

            for entry in feed.entries:
                text = clean_text(entry.summary)

                if not text:
                    continue
                if not is_english(text):
                    continue
                if text in seen:
                    continue

                seen.add(text)
                results.append({
                    "source": "twitter_rss",
                    "text": text
                })

                if len(results) >= limit:
                    return results
        except Exception as e:
            log("Error:", e)
            continue

    log("Final count:", len(results))
    return results
