import pandas as pd
import numpy as np
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Load dataset
df = pd.read_csv("fake_news_dataset.csv")

# Clean missing values
df = df.dropna(subset=['title', 'text'])

# Target variable
# Map 'Fake' → 1, 'Real' → 0 (or vice versa depending on your target)
df['label'] = df['label'].map({'Fake': 1, 'Real': 0})


# ----- TEXT + METADATA FEATURE PIPELINE -----

# Combine title and text
df['full_text'] = df['title'] + " " + df['text']

# Custom transformer for metadata
class MetaFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, meta_cols):
        self.meta_cols = meta_cols
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.meta_cols])
        return self

    def transform(self, X):
        return self.scaler.transform(X[self.meta_cols])

# Select metadata features
meta_features = [
    'sentiment_score', 'word_count', 'char_count', 'has_images', 'has_videos',
    'readability_score', 'num_shares', 'num_comments',
    'political_bias', 'fact_check_rating', 'is_satirical',
    'trust_score', 'source_reputation', 'clickbait_score', 'plagiarism_score'
]

# Fill NA values in metadata
df[meta_features] = df[meta_features].fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df, df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer for text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Transform full_text
X_train_text = vectorizer.fit_transform(X_train['full_text'])
X_test_text = vectorizer.transform(X_test['full_text'])

# Metadata Transformer
meta_transformer = MetaFeaturesExtractor(meta_cols=meta_features)
X_train_meta = meta_transformer.fit_transform(X_train)
X_test_meta = meta_transformer.transform(X_test)

# Combine text and metadata
from scipy.sparse import hstack
X_train_combined = hstack([X_train_text, X_train_meta])
X_test_combined = hstack([X_test_text, X_test_meta])

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_combined, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_combined)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "text_vectorizer.pkl")
joblib.dump(meta_transformer, "meta_transformer.pkl")
print("\n💾 Model, vectorizer, and meta transformer saved successfully!")
