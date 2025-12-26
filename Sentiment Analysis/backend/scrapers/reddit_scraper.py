import requests
from .helpers import clean_text, is_english

USER_AGENT = "Mozilla/5.0 (compatible; BizChainAI/1.0)"

def scrape_reddit(query, limit=120, debug=False):

    def log(*m):
        if debug:
            print("[REDDIT SCRAPER]", *m)

    url = f"https://www.reddit.com/search.json?q={query}&limit=100"

    headers = {"User-Agent": USER_AGENT}

    try:
        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code != 200:
            log("Error:", r.text)
            return []

        data = r.json()
        posts = data.get("data", {}).get("children", [])

        results = []
        seen = set()

        for p in posts:
            title = p["data"].get("title", "")
            selftext = p["data"].get("selftext", "")

            text = clean_text(title + " " + selftext)

            if not text:
                continue
            if not is_english(text):
                continue
            if text in seen:
                continue

            seen.add(text)

            results.append({
                "source": "reddit",
                "text": text
            })

            if len(results) >= limit:
                break

        log(f"Collected {len(results)} Reddit posts.")
        return results

    except Exception as e:
        log("Exception:", e)
        return []
