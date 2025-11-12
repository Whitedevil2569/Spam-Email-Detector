import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("emails.csv", encoding="cp1252")
df["text"] = df["Subject"].fillna('') + " " + df["Body"].fillna('')
df = df.rename(columns={"Spam Label": "label"})

X = df["text"]
y = df["label"]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)


joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(accuracy, "models/accuracy.pkl")