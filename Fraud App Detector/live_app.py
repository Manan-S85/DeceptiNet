from flask import Flask, render_template_string, request
from google_play_scraper import search, app, reviews
from textblob import TextBlob
import pandas as pd
import joblib
import re

app_flask = Flask(__name__)

# Load model and fraud list
model = joblib.load('fraud_app_model10.pkl')
fraud_titles = pd.read_excel("fraud apps.xlsx")['App name'].str.strip().str.lower().tolist()

feature_columns = [
    'Rating', 'Installs', 'Reviews', 'review_length', 'exclamations',
    'all_caps_count', 'sentiment_polarity', 'sentiment_subjectivity'
]

def clean_installs(value):
    value = re.sub(r'[+,]', '', value)
    return int(value) if value.isdigit() else 0

def clean_price(value):
    if isinstance(value, str) and value.startswith('$'):
        return float(value.replace('$', ''))
    return 0.0

def extract_review_features(reviews_list):
    contents = [r['content'] for r in reviews_list if isinstance(r, dict) and 'content' in r]
    df = pd.DataFrame({'content': contents})
    df['review_length'] = df['content'].apply(lambda x: len(str(x).split()))
    df['exclamations'] = df['content'].apply(lambda x: str(x).count('!'))
    df['all_caps_count'] = df['content'].apply(lambda x: sum(1 for w in str(x).split() if w.isupper()))
    df['sentiment_polarity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['sentiment_subjectivity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    return df.mean(numeric_only=True).to_dict()

@app_flask.route("/", methods=["GET", "POST"])
def index():
    result = None
    app_title = ""
    error = None

    if request.method == "POST":
        user_input = request.form["appname"].strip().lower()

        if user_input in fraud_titles:
            result = "🚨 FRAUDULENT (Listed in known fraud apps)"
            app_title = user_input
        else:
            try:
                results = search(user_input)
                if not results:
                    error = "App not found on Play Store."
                else:
                    best_match = results[0]
                    pkg_name = best_match['appId']
                    title_found = best_match['title'].strip().lower()

                    app_info = app(pkg_name)
                    app_title = app_info.get('title', '')
                    rating = app_info.get('score', 0.0)
                    installs = clean_installs(app_info.get('installs', '0'))
                    reviews_count = int(app_info.get('reviews', 0))

                    review_list, _ = reviews(pkg_name, count=100)
                    review_features = extract_review_features(review_list)

                    data = {
                        'Rating': rating,
                        'Installs': installs,
                        'Reviews': reviews_count,
                        'review_length': review_features.get('review_length', 0),
                        'exclamations': review_features.get('exclamations', 0),
                        'all_caps_count': review_features.get('all_caps_count', 0),
                        'sentiment_polarity': review_features.get('sentiment_polarity', 0),
                        'sentiment_subjectivity': review_features.get('sentiment_subjectivity', 0),
                    }

                    df = pd.DataFrame([data])[feature_columns]
                    prediction = model.predict(df)[0]
                    result = "🚨 FRAUDULENT" if prediction == 1 else "✔️ NOT FRAUDULENT"
            except Exception as e:
                error = f"Something went wrong: {str(e)}"

    return render_template_string(html_template, result=result, title=app_title, error=error)

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fraud App Detector | ClickSafe</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: #fff;
            margin: 0;
            padding: 0;
            text-align: center;
        }

        .container {
            max-width: 500px;
            margin: 100px auto;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        }

        h1 {
            margin-bottom: 10px;
        }

        input[type="text"] {
            width: 80%;
            padding: 12px;
            border: none;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 16px;
        }

        button {
            padding: 12px 24px;
            font-size: 16px;
            background: #1abc9c;
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            margin-top: 10px;
        }

        button:hover {
            background: #16a085;
        }

        .home-btn {
            background: #3498db;
        }

        .home-btn:hover {
            background: #2980b9;
        }

        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 12px;
            background-color: rgba(255, 255, 255, 0.1);
            font-size: 18px;
        }

        .label {
            font-size: 22px;
            font-weight: bold;
            margin-top: 10px;
        }

        .error {
            color: #ff6b6b;
            margin-top: 15px;
        }
    </style>
</head>
<body>

    <div class="container">
        <h1>Fraud App Detector 🔍</h1>
        <p>Enter a Play Store app name to find out if it's fake or safe.</p>
        
        <form method="POST">
            <input type="text" name="appname" placeholder="Enter app name..." required>
            <br>
            <button type="submit">Check App</button>
        </form>

        <a href="http://127.0.0.1:5500/Codebase/index.html">
            <button type="button" class="home-btn">🏠 Home</button>
        </a>

        {% if result %}
            <div class="result">
                <div><strong>App:</strong> {{ title }}</div>
                <div class="label">{{ result }}</div>
            </div>
        {% endif %}

        {% if error %}
            <div class="error">⚠️ {{ error }}</div>
        {% endif %}
    </div>

</body>
</html>
"""

if __name__ == "__main__":
    app_flask.run(debug=True, port=5000)
