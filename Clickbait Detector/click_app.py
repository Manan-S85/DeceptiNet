from flask import Flask, render_template_string, request
import joblib

app_clickbait = Flask(__name__)

# Load model and vectorizer
model = joblib.load("clickbait_model.pkl")
vectorizer = joblib.load("clickbait_vectorizer.pkl")

def extract_features(text):
    return vectorizer.transform([text])

@app_clickbait.route("/", methods=["GET", "POST"])
def index():
    result = None
    headline = ""

    if request.method == "POST":
        headline = request.form["headline"].strip()
        if headline:
            features = extract_features(headline)
            prediction = model.predict(features)[0]
            result = "🚨 Clickbait Detected!" if prediction == 1 else "✅ Not Clickbait."

    return render_template_string(html_template, result=result, headline=headline)

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Clickbait Detector | ClickSafe</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #141e30, #243b55);
            color: #fff;
            margin: 0;
            padding: 0;
            text-align: center;
        }

        .container {
            max-width: 600px;
            margin: 100px auto;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 40px 30px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.4);
        }

        h1 {
            margin-bottom: 10px;
        }

        p {
            font-size: 16px;
            color: #ddd;
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
            background: #e67e22;
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
        }

        button:hover {
            background: #d35400;
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
    </style>
</head>
<body>

    <div class="container">
        <h1>Clickbait Headline Detector 🎯</h1>
        <p>Enter any news headline to check if it’s clickbait or not.</p>

        <form method="POST">
            <input type="text" name="headline" placeholder="Type headline here..." required>
            <br>
            <button type="submit">Check Headline</button>
        </form>
        
        <div style="margin-top: 15px;">
    <a href="http://127.0.0.1:5500/Codebase/index.html">
        <button type="button" class="home-btn">🏠 Home</button>
    </a>
</div>

        {% if result %}
            <div class="result">
                <div><strong>Headline:</strong> {{ headline }}</div>
                <div class="label">{{ result }}</div>
            </div>
        {% endif %}
    </div>

</body>
</html>
"""

if __name__ == "__main__":
    app_clickbait.run(debug=True, port=5001)

