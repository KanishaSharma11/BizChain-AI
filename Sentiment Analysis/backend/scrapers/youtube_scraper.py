# youtube_scraper.py

from googleapiclient.discovery import build
from .helpers import clean_text, is_english

API_KEY = "AIzaSyCSAYH72FlkIZJh-H4ZHRBQhT4b6c9o47M"

# Words indicating Indian creators (skip these)
INDIAN_KEYWORDS = [
    "india", "hindi", "indian", "bharat", "delhi", "mumbai", "punjab",
    "tamil", "telugu", "kannada", "malayalam", "desi", "bollywood"
]

# Acceptable foreign English audio languages
FOREIGN_AUDIO = ["en-US", "en-GB", "en-AU", "en-CA"]

def is_indian_text(text):
    text = text.lower()
    return any(word in text for word in INDIAN_KEYWORDS)

def scrape_youtube(query, limit=50):
    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)

        # 1️⃣ Search videos
        search = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=10,
            order="relevance"
        ).execute()

        if not search.get("items"):
            print("[YouTube] No results.")
            return []

        video_ids = [item["id"]["videoId"] for item in search["items"]]

        # 2️⃣ Fetch stats + snippet for language + channel country
        stats = youtube.videos().list(
            part="statistics,snippet",
            id=",".join(video_ids)
        ).execute()

        best_video_id = None
        best_views = -1

        for video in stats.get("items", []):
            snippet = video["snippet"]
            stats_info = video["statistics"]

            title = snippet["title"].lower()
            description = snippet.get("description", "").lower()
            channel_title = snippet.get("channelTitle", "").lower()

            country = snippet.get("defaultLanguage", "")
            audio = snippet.get("defaultAudioLanguage", "")
            views = int(stats_info.get("viewCount", 0))

            # 2A: Reject videos with Indian signals
            if is_indian_text(title) or is_indian_text(description) or is_indian_text(channel_title):
                continue

            # 2B: Reject Indian English audio
            if audio == "en-IN":
                continue

            # 2C: Reject Hindi or Indian language
            if audio.startswith("hi"):
                continue

            # 2D: Accept only foreign English audio if available
            if audio and audio not in FOREIGN_AUDIO:
                continue

            # Keep the highest-view foreign English video
            if views > best_views:
                best_views = views
                best_video_id = video["id"]

        if not best_video_id:
            print("[YouTube] No foreign English video found.")
            return []

        print(f"[YouTube] Selected foreign creator video → {best_video_id} ({best_views} views)")

        # 3️⃣ Fetch comments
        comments_raw = youtube.commentThreads().list(
            part="snippet",
            videoId=best_video_id,
            maxResults=100,
            textFormat="plainText"
        ).execute()

        results = []

        for item in comments_raw.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            text = clean_text(text)

            if not text:
                continue
            if not is_english(text):
                continue

            results.append({
                "source": "youtube",
                "text": text
            })

            if len(results) >= limit:
                break

        print(f"[YouTube] Extracted {len(results)} English comments")
        return results

    except Exception as e:
        print("YouTube Scraper Error:", e)
        return []
