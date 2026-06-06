import pandas as pd
df=pd.read_csv(r"E:\Customer_Feedback_Sentiment_Analyzer\Data\cleaned_imdb_dataset.csv")
print("First 5 data(rows)")
print(df.head())
print("Type: ",type(df.head()))
print("Dimension row and column: ",df.shape)
print(df["sentiment"].value_counts())
print(df["review"][0])
