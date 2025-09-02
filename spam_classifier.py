import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
df.columns = ["label", "text"]

# Step 2: Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

# Step 4: Convert text to feature vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train a classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test_vec)

# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Test with custom input
while True:
    user_input = input("\nEnter a message to classify (or 'exit'): ")
    if user_input.lower() == 'exit':
        break
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    print("Prediction:", "Spam" if prediction == 1 else "Ham (Not Spam)")
