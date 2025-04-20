# model.py
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# -------------------------------
# Load datasets
# -------------------------------
scraped_apps = pd.read_csv('app_dataset_full.csv')
reviews_df = pd.read_csv('review_dataset_full.csv')
fraud_apps = pd.read_excel('fraud apps.xlsx')  # 100% fraud apps

# -------------------------------
# Standardize names
# -------------------------------
scraped_apps['Title_clean'] = scraped_apps['Title'].str.strip().str.lower()
fraud_apps['App_clean'] = fraud_apps['App name'].str.strip().str.lower()

# -------------------------------
# Assign fraud labels
# -------------------------------
scraped_apps['fraud_label'] = 0  # All scraped apps are legit
scraped_apps['App_clean'] = scraped_apps['Title_clean']

fraud_apps_df = pd.DataFrame({
    'App_clean': fraud_apps['App_clean'],
    'fraud_label': 1
})

# Merge all apps on App_clean
combined_apps = pd.concat([
    scraped_apps[['App_clean', 'Title', 'Rating', 'Installs', 'Reviews', 'Category', 'Price', 'Content Rating', 'fraud_label']],
    fraud_apps_df
], ignore_index=True)

print("🔢 Fraud label distribution:\n", combined_apps['fraud_label'].value_counts())

# -------------------------------
# Process review dataset
# -------------------------------
def extract_review_features(df):
    df['review_length'] = df['content'].apply(lambda x: len(str(x).split()))
    df['exclamations'] = df['content'].apply(lambda x: str(x).count('!'))
    df['all_caps_count'] = df['content'].apply(lambda x: sum(1 for word in str(x).split() if word.isupper()))
    df['sentiment_polarity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['sentiment_subjectivity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    return df

print("🛠️ Extracting review features...")
reviews_df = extract_review_features(reviews_df)

# Standardize for merging
reviews_df['App_clean'] = reviews_df['App'].str.strip().str.lower()

# -------------------------------
# Aggregate Review Features Per App
# -------------------------------
print("📊 Aggregating per app...")
app_review_features = reviews_df.groupby('App_clean').agg({
    'review_length': 'mean',
    'exclamations': 'mean',
    'all_caps_count': 'mean',
    'sentiment_polarity': 'mean',
    'sentiment_subjectivity': 'mean'
}).reset_index()

# -------------------------------
# Merge with app metadata
# -------------------------------
combined_apps = combined_apps.merge(app_review_features, on='App_clean', how='left')

# -------------------------------
# Clean columns
# -------------------------------
combined_apps['Installs'] = combined_apps['Installs'].astype(str).str.replace('[+,]', '', regex=True)
combined_apps['Installs'] = pd.to_numeric(combined_apps['Installs'], errors='coerce')
combined_apps['Reviews'] = pd.to_numeric(combined_apps['Reviews'], errors='coerce')

# -------------------------------
# Train-Test Split and Model
# -------------------------------
features = ['Rating', 'Installs', 'Reviews', 'review_length', 'exclamations',
            'all_caps_count', 'sentiment_polarity', 'sentiment_subjectivity']
combined_apps[features] = combined_apps[features].fillna(0)

X = combined_apps[features]
y = combined_apps['fraud_label']

print("🤖 Training Random Forest Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Evaluation and Saving
# -------------------------------
y_pred = model.predict(X_test)
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, 'fraud_app_model10.pkl')
print("💾 Model saved to 'fraud_app_model7.pkl'")

# Save for prediction use (include Title to match from UI)
combined_apps.to_csv('final_app_features4.csv', index=False)
print("📦 Feature data saved to 'final_app_features.csv'")