#importing 
import nltk
import string
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


#data
reviews = [
    "Amazing product",
    "Worst experience",
    "Very good quality",
    "Not worth the money",
    "Excellent service",
    "Terrible support",
    "Loved it",
    "Very bad item"
]

sentiments = [
    "Positive",
    "Negative",
    "Positive",
    "Negative",
    "Positive",
    "Negative",
    "Positive",
    "Negative"
]
X = reviews
y = sentiments

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

"""
| Step              | What Happens                                       |
| ----------------- | -------------------------------------------------- |
| `fit()`           | Learn vocabulary + IDF statistics                  |
| `fit_transform()` | Learn + convert training text into vectors         |
| `transform()`     | Convert new text using existing learned vocabulary |
"""

#Train Logistic Regression Model

model = LogisticRegression()
model.fit(X_train_tfidf,y_train)

#Make Predictions
y_pred = model.predict(X_test_tfidf)


#Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


#Custom Prediction
new_review = ["This product is really amazing"]

new_review_tfidf = vectorizer.transform(new_review)

prediction = model.predict(new_review_tfidf)

print("Prediction:", prediction[0])


print("X_test:", X_test)
print("Actual:", y_test)
print("Predicted:", y_pred)