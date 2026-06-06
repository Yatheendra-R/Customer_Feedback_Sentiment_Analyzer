import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import joblib

# Load data
df = pd.read_csv(r"E:\Customer_Feedback_Sentiment_Analyzer\Data\cleaned_imdb_lstm_td.csv")

X = df["review"]
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# TF-IDF (optimized)
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),
    max_features=20000,
    min_df=2,
    max_df=0.85,
    sublinear_tf=True,
    smooth_idf=True

)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# MODEL: Linear SVM
model =LinearSVC(C=0.5)
#It controls how much the model should care about misclassifying training data.
"""
LinearSVC tries to find a decision boundary (line/hyperplane).

It balances two things:

1. Maximize margin (generalization)
2. Minimize classification errors

| C value                  | Behavior              | Effect                                      |
| ------------------------ | --------------------- | ------------------------------------------- |
| Small C (0.01, 0.1, 0.5) | strict regularization | model is more simple, ignores some mistakes |
| Medium C (1.0)           | balanced              | good generalization                         |
| Large C (10, 100)        | weak regularization   | tries to fit training data very closely     |

"""
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.title("Confusion Matrix - LinearSVC + TF-IDF")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

"""
output:
Accuracy: 0.9115

Classification Report:
               precision    recall  f1-score   support

    negative       0.92      0.91      0.91      4961
    positive       0.91      0.92      0.91      5039

    accuracy                           0.91     10000
   macro avg       0.91      0.91      0.91     10000
weighted avg       0.91      0.91      0.91     10000
"""
# Save model

joblib.dump(model, r"E:\Customer_Feedback_Sentiment_Analyzer\Saved\Tonedownlemma\SVC\Saved_sentiment_model_svm.pkl")

# Save TF-IDF vectorizer
joblib.dump(vectorizer, r"E:\Customer_Feedback_Sentiment_Analyzer\Saved\Tonedownlemma\SVC\Saved_tfidf_vectorizer.pkl")


#save config
config = {
    "ngram_range": (1,3),
    "max_features": 20000,
    "min_df": 2,
    "max_df": 0.85,
    "C": 0.5
}

joblib.dump(config, r"E:\Customer_Feedback_Sentiment_Analyzer\Saved\Tonedownlemma\SVC\config.pkl")
print("Model and vectorizer saved successfully!")