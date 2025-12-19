import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# 1. Load your existing dataset
data = pd.read_csv('data/careers.csv')

# 2. Separate columns
X = data['text']
y = data['career']

# 3. Convert text to numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# 4. Train the model
model = MultinomialNB()
model.fit(X_vec, y)

# 5. Save the model
with open('model/recommender.pkl', 'wb') as f:
    pickle.dump(model, f)

# 6. Save the vectorizer
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")