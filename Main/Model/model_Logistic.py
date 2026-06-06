import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

df=pd.read_csv(r"E:\Customer_Feedback_Sentiment_Analyzer\Data\cleaned_imdb_dataset.csv")


X=df["review"]
y=df["sentiment"]

#Train and Testing split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#vectorizing object
vectorizer =TfidfVectorizer() 

#Train vocabulary + transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)

#Transform test data only
X_test_tfidf=vectorizer.transform(X_test)

#Training model
model = LogisticRegression()

model.fit(X_train_tfidf, y_train)


#Predictions

y_pred = model.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("classification_report: ",classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

"""
RESULT:

For:

    TF-IDF
    Logistic Regression
    Basic preprocessing

89% is a strong baseline result 

|            | Predicted Neg  | Predicted Pos  |
| ---------- | -------------- | -------------- |
| Actual Neg | 4360 (correct) | 601 (wrong)    |
| Actual Pos | 460 (wrong)    | 4579 (correct) |

"""