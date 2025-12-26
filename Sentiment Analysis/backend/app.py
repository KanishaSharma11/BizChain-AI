from flask import Flask, request, jsonify, send_file, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from datetime import datetime
import os

# -------------------------
# Scrapers
# -------------------------
from scrapers.twitter_scraper import scrape_twitter
from scrapers.news_scraper import scrape_news
from scrapers.youtube_scraper import scrape_youtube
from scrapers.reddit_scraper import scrape_reddit

app = Flask(__name__, template_folder="templates")

# -------------------------
# Load Trained BERT Model
# -------------------------
tokenizer = BertTokenizer.from_pretrained("sentiment_model")
model = BertForSequenceClassification.from_pretrained("sentiment_model")

with open("sentiment_model/label_encoder.pkl", "rb") as f:
    label_encoder = joblib.load(f)


# -------------------------
# Home Route
# -------------------------
@app.route("/")
def home():
    return render_template("sentiment.html")


# -------------------------
# Predict Sentiment
# -------------------------
def predict_sentiment(text):

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if text == "":
        return "neutral"

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits).item()

    return label_encoder.inverse_transform([pred])[0]


# ============================================================
#    UNIFIED SCRAPE + ANALYZE PIPELINE
# ============================================================
def scrape_all_sources(query, sources, video_id):
    results = []

    print("\n--- SCRAPING SOURCES ---")
    print("Query:", query)
    print("Sources:", sources)

    # -----------------------------------
    # TWITTER (if enabled)
    # -----------------------------------
    if "twitter" in sources:
        try:
            print("Scraping Twitter...")
            tweets = scrape_twitter(query, limit=50)
            for t in tweets:
                results.append({"source": "Twitter", "text": t["text"]})
            print("Twitter returned", len(tweets), "items")
        except Exception as e:
            print("Twitter Error:", e)

    # -----------------------------------
    # NEWS (works great)
    # -----------------------------------
    if "news" in sources:
        try:
            print("Scraping News...")
            news_items = scrape_news(query, limit=20)
            for n in news_items:
                results.append({"source": "News", "text": n["text"]})
            print("News returned", len(news_items), "items")
        except Exception as e:
            print("News Error:", e)

    # -----------------------------------
    # YOUTUBE COMMENTS (ADDED NOW)
    # -----------------------------------
    if "youtube" in sources:
        try:
            print("Scraping YouTube...")
            yt_items = scrape_youtube(query, limit=50)
            for y in yt_items:
                results.append({"source": "YouTube", "text": y["text"]})
            print("YouTube returned", len(yt_items), "items")
        except Exception as e:
            print("YouTube Error:", e)

    # -----------------------------------
    # REDDIT (NEW)
    # -----------------------------------
    if "reddit" in sources:
        try:
            print("Scraping Reddit...")
            reddit_items = scrape_reddit(query, limit=50)
            for r in reddit_items:
                results.append({"source": "Reddit", "text": r["text"]})
            print("Reddit returned", len(reddit_items), "items")
        except Exception as e:
            print("Reddit Error:", e)
        

    print("--- SCRAPING DONE ---")
    print("Total collected:", len(results))

    return results


# ============================================================
#    ANALYZE COMMENTS
# ============================================================
def analyze_comments(scraped):
    analyzed = []

    for item in scraped:
        text = item.get("text", "")

        if not isinstance(text, str) or text.strip() == "":
            continue

        sentiment = predict_sentiment(text)

        analyzed.append({
            "source": item["source"],
            "text": text,
            "sentiment": sentiment
        })

    return analyzed


# ============================================================
#    /analyze ENDPOINT
# ============================================================
@app.post("/analyze")
def analyze():
    data = request.json
    query = data.get("query")
    sources = data.get("sources", [])
    video_id = data.get("video_id")

    scraped = scrape_all_sources(query, sources, video_id)
    analyzed = analyze_comments(scraped)

    summary = {
        "total": len(analyzed),
        "positive": sum(1 for r in analyzed if r["sentiment"] == "positive"),
        "neutral": sum(1 for r in analyzed if r["sentiment"] == "neutral"),
        "negative": sum(1 for r in analyzed if r["sentiment"] == "negative")
    }

    return jsonify({
        "summary": summary,
        "comments": analyzed
    })


# ============================================================
#    PDF GENERATOR
# ============================================================
def generate_pdf(query, comments, summary):
    folder = "generated_reports"
    os.makedirs(folder, exist_ok=True)

    filename = f"{folder}/{query}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf = SimpleDocTemplate(filename, pagesize=A4)

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>BizChain AI - Sentiment Report for {query}</b>", styles["Title"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph(f"Total Comments: {summary['total']}", styles["Normal"]))
    story.append(Paragraph(f"Positive: {summary['positive']}", styles["Normal"]))
    story.append(Paragraph(f"Neutral: {summary['neutral']}", styles["Normal"]))
    story.append(Paragraph(f"Negative: {summary['negative']}", styles["Normal"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph("<b>Comments:</b>", styles["Heading2"]))

    for c in comments:
        story.append(Paragraph(
            f"<b>{c['sentiment'].upper()}</b> — {c['source']} — {c['text']}", styles["BodyText"]
        ))
        story.append(Spacer(1, 10))

    pdf.build(story)
    return filename


# ============================================================
#    /download-pdf ENDPOINT
# ============================================================
@app.post("/download-pdf")
def download_pdf():
    data = request.json
    path = generate_pdf(
        data["query"],
        data["comments"],
        data["summary"]
    )
    return send_file(path, download_name="BizChainAI_Report.pdf", as_attachment=True)


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    app.run(port=5005, debug=True)
