import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("clickbait.csv")  # Replace with your actual filename

# Features and target
X = df['text'].astype(str)  # Ensure all are strings
y = df['label']

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

# Save model and vectorizer
joblib.dump(model, "clickbait_model.pkl")
joblib.dump(vectorizer, "clickbait_vectorizer.pkl")
print("\n💾 Model and vectorizer saved successfully!")
