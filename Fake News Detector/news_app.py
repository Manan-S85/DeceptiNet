from flask import Flask, render_template_string, jsonify
import feedparser
from transformers import pipeline
from datetime import datetime

app_fake_news = Flask(__name__)

classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
RSS_FEED_URL = "https://news.google.com/rss?hl=en-GB&gl=GB&ceid=GB:en"

def fetch_and_classify():
    feed = feedparser.parse(RSS_FEED_URL)
    results = []
    for entry in feed.entries[:15]:
        title = entry.title
        link = entry.link
        try:
            result = classifier(title)[0]
            raw_label = result["label"]
            score = round(result["score"] * 100, 1)
            if raw_label == "LABEL_1":
                label = "FAKE"
            elif raw_label == "LABEL_0":
                label = "REAL"
            else:
                label = "UNKNOWN"
        except:
            label, score = "ERROR", 0.0
        results.append({"title": title, "label": label, "score": score, "link": link})
    return results

@app_fake_news.route("/")
def index():
    news_results = fetch_and_classify()
    return render_template_string(html_template, news=news_results)

@app_fake_news.route("/news")
def news_only():
    news_results = fetch_and_classify()
    return render_template_string(news_card_template, news=news_results)

news_card_template = """
{% for item in news %}
<div class="card">
    <h2>{{ item.title }}</h2>
    <div class="label {{ item.label }}">{{ item.label }} — {{ item.score }}%</div>
    <div class="confidence"><a href="{{ item.link }}" target="_blank">Read full story 🔗</a></div>
</div>
{% endfor %}
"""

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fake News Live Detector | ClickSafe</title>
    <style>
        body {
            background: linear-gradient(120deg, #1f1c2c, #928DAB);
            font-family: 'Segoe UI', sans-serif;
            color: #fff;
            padding: 40px;
        }

        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        p {
            text-align: center;
            font-size: 1.1rem;
            color: #ddd;
            margin-bottom: 20px;
        }

        .home-btn-container {
            text-align: center;
            margin-bottom: 40px;
        }

        .home-btn {
            background: #f39c12;
            border: none;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            color: white;
            text-decoration: none;
        }

        .home-btn:hover {
            background: #d35400;
        }

        .card {
            background: rgba(255, 255, 255, 0.08);
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 12px;
            box-shadow: 0 0 12px rgba(0,0,0,0.2);
        }

        .card h2 {
            font-size: 1.2rem;
            margin-bottom: 5px;
        }

        .card a {
            color: #00acee;
            font-size: 0.95rem;
            text-decoration: none;
        }

        .label {
            font-weight: bold;
            font-size: 1rem;
            margin-top: 8px;
        }

        .FAKE {
            color: #e74c3c;
        }

        .REAL {
            color: #2ecc71;
        }

        .UNKNOWN, .ERROR {
            color: #f39c12;
        }

        .confidence {
            font-size: 0.9rem;
            color: #bbb;
        }
    </style>
</head>
<body>
    <h1>📰 Live Fake News Detector</h1>
    <p>Checking top global news headlines in real time using AI.</p>

    <div class="home-btn-container">
        <a href="http://127.0.0.1:5500/Codebase/index.html" class="home-btn">🏠 Home</a>
    </div>

    <div id="news-container">
        {% for item in news %}
        <div class="card">
            <h2>{{ item.title }}</h2>
            <div class="label {{ item.label }}">{{ item.label }} — {{ item.score }}%</div>
            <div class="confidence"><a href="{{ item.link }}" target="_blank">Read full story 🔗</a></div>
        </div>
        {% endfor %}
    </div>

    <script>
        function fetchNews() {
            fetch("/news")
                .then(response => response.text())
                .then(html => {
                    document.getElementById("news-container").innerHTML = html;
                });
        }

        // Refresh every 30 seconds
        setInterval(fetchNews, 30000);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    app_fake_news.run(debug=True, port=5002)
