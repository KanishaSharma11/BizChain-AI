import requests
import random
import concurrent.futures
from bs4 import BeautifulSoup
from urllib.parse import quote
from difflib import SequenceMatcher
from .helpers import clean_text, is_english, HEADERS

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)",
]

def get_headers():
    h = HEADERS.copy()
    h["User-Agent"] = random.choice(USER_AGENTS)
    return h


# ------------------------------
# Utility: Fetch URL
# ------------------------------
def fetch(url):
    try:
        r = requests.get(url, headers=get_headers(), timeout=10)
        if r.status_code == 200:
            return r.text
    except:
        return None
    return None


# ------------------------------
# Utility: Extract OG Description
# ------------------------------
def extract_summary(html):
    try:
        soup = BeautifulSoup(html, "html.parser")

        # OpenGraph
        meta = soup.find("meta", property="og:description")
        if meta and meta.get("content"):
            return clean_text(meta["content"])

        # Regular meta
        meta = soup.find("meta", {"name": "description"})
        if meta and meta.get("content"):
            return clean_text(meta["content"])

    except:
        return ""
    return ""


# ------------------------------
# Utility: Deduplicate by title similarity
# ------------------------------
def is_similar(t1, t2, threshold=0.8):
    return SequenceMatcher(None, t1.lower(), t2.lower()).ratio() > threshold


# ------------------------------
# MAIN OPTIMIZED NEWS SCRAPER
# ------------------------------
def scrape_news(query, limit=150, debug=False):

    query_lower = query.lower()
    results = []
    seen_urls = set()
    seen_titles = []

    def log(*msg):
        if debug:
            print("[NEWS]", *msg)

    # -------------------------------------------
    # 1. Build all Google & Bing URLs first
    # -------------------------------------------
    rss_windows = [
        "", "when%3A1h", "when%3A3h", "when%3A6h", "when%3A1d",
        "when%3A3d", "when%3A7d", "when%3A30d"
    ]

    urls = []

    # Google RSS windows
    for win in rss_windows:
        urls.append(
            f"https://news.google.com/rss/search?q={quote(query)}+{win}&hl=en-IN&gl=IN&ceid=IN:en"
        )

    # Google HTML pages
    for start in [0, 10, 20, 30, 40]:
        urls.append(
            f"https://www.google.com/search?q={quote(query)}&tbm=nws&start={start}"
        )

    # Bing + Yahoo
    urls.append(f"https://www.bing.com/news/search?q={quote(query)}&format=rss")
    urls.append(f"https://news.search.yahoo.com/rss?p={quote(query)}")

    # -------------------------------------------
    # 2. Fetch all URLs in parallel (fast)
    # -------------------------------------------
    log("Fetching URLs in parallel:", len(urls))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        html_pages = list(executor.map(fetch, urls))

    # -------------------------------------------
    # 3. Parse each page
    # -------------------------------------------
    for html in html_pages:
        if not html:
            continue

        soup = BeautifulSoup(html, "xml") if "<rss" in html else BeautifulSoup(html, "html.parser")

        # --- Parse RSS items ---
        items = soup.find_all("item")
        for it in items:
            title = clean_text(it.title.text if it.title else "")
            url = it.link.text if it.link else ""

            if not title or not url:
                continue
            if not is_english(title):
                continue

            # Relevance filter (must contain at least one query keyword)
            if not any(word in title.lower() for word in query_lower.split()):
                continue

            # Dedup URL
            if url in seen_urls:
                continue

            # Dedup similar titles
            if any(is_similar(title, t) for t in seen_titles):
                continue

            seen_urls.add(url)
            seen_titles.append(title)

            results.append({
                "source": "rss",
                "title": title,
                "text": title,
                "url": url,
            })

            if len(results) >= limit:
                return results

        # --- Parse Google HTML Blocks ---
        blocks = soup.select("div.dbsr") or soup.select("g-card")

        for b in blocks:
            a = b.select_one("a")
            title_node = b.select_one("div.JheGif") or b.select_one("div.nDgy9d")

            if not a or not title_node:
                continue

            url = a.get("href")
            title = clean_text(title_node.get_text(strip=True))

            if not is_english(title):
                continue
            if not any(word in title.lower() for word in query_lower.split()):
                continue
            if url in seen_urls:
                continue
            if any(is_similar(title, t) for t in seen_titles):
                continue

            # Fetch article summary
            article_html = fetch(url)
            summary = extract_summary(article_html) if article_html else title

            seen_urls.add(url)
            seen_titles.append(title)

            results.append({
                "source": "google-html",
                "title": title,
                "text": summary,
                "url": url,
            })

            if len(results) >= limit:
                return results

    log("FINAL COUNT:", len(results))
    return results
