# 🚨 DeceptiNet: Detecting Digital Deception
DeceptiNet is a unified AI-powered platform to detect fraudulent apps, clickbait headlines, and fake news in real-time. This repository integrates all three detection models and a simple web interface to interact with them live.

You can explore the deployed version of the website here (https://manan-s85.github.io/DeceptiNet/).


# 📂 Repository Structure
# 🔍 fraud_app_detection/
A complete pipeline for detecting potentially fraudulent Android apps from the Google Play Store.

Dataset: Includes preprocessed data with metadata, user reviews, and sentiment analysis.

Model Code: Machine learning models trained on behavioral and sentiment-based features.

live_app.py: Live predictor that uses GooglePlayStoreScraper to fetch app data and run real-time fraud checks.

Usage: Enter any app name to instantly check if it’s suspicious based on reviews, install patterns, and ratings.

# 📰 clickbait_detection/
A system to flag clickbait headlines using NLP and statistical text features.

Dataset: Includes labeled news headlines for training and validation.

Model Code: Feature extraction + classification using machine learning (TF-IDF, NLP features, etc.).

click_app.py: Live prediction script to classify any input headline as clickbait or not.

Usage: Paste any news headline and instantly detect whether it’s misleading or not.

# 🌍 fake_news_updater/
A live fake news detection system that updates every 5 minutes using real-time news scraping.

Model: Pretrained text classification model that detects fake news from trusted sources.

news_app.py: Background script that fetches fresh headlines (via GNews/RSS) and classifies them automatically.

Auto-refresh: Keeps updating new results periodically for continuous monitoring.

Usage: Automatically monitors the web and detects misinformation in real time.

# 🌐 website/
A simple web interface to access and run all three detection models from one place.

Built with basic HTML & internal CSS for lightweight performance.

Each model runs through its respective live.py in the background when launched.

No complex backend—just open in your browser and use!
