#################################################
#importing 

import string 
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

########################################


################Clean TXT function##################
def clean_text(text):

    #To lower case
    text=text.lower()

    #remove HTML tags
    text=re.sub(r"<.*?>"," ",text)
    
    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    """
    | Regex      | Meaning                           |
    | ---------- | --------------------------------- |
    | `\s+`      | one or more whitespace characters |
    | `" "`      | replace with single space         |
    | `.strip()` | remove extra spaces at start/end  |

    """

    #replacing(Expand) imporatant term which has punctuation
    contractions = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "can't": "can not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "never": "not ever"
    }

    for short_form,full_form in contractions.items():
        text=text.replace(short_form,full_form)
    
    #remove punctuation
    text=text.translate(str.maketrans("","",string.punctuation))

    #List of word
    tokens=text.split()

    #Set of stopwords
    stop_words=set(stopwords.words("English"))

    #import words present in stopwords
    important_words = [
    "not",
    "no",
    "nor",
    "never",

    "but",

    "very",

    "should",
    "should've",

    "couldn",
    "couldn't",

    "wouldn",
    "wouldn't",

    "won",
    "won't",

    "don",
    "don't",

    "didn",
    "didn't",

    "doesn",
    "doesn't",

    "hadn",
    "hadn't",

    "hasn",
    "hasn't",

    "haven",
    "haven't",

    "isn",
    "isn't",

    "aren",
    "aren't",

    "wasn",
    "wasn't",

    "weren",
    "weren't",

    "mightn",
    "mightn't",

    "mustn",
    "mustn't",

    "needn",
    "needn't",

    "shan",
    "shan't"

    ]


    #Removing important words from stop words
    for imp_word in important_words:
        if imp_word in stop_words:
            stop_words.discard(imp_word)


    #Removing stop word from the tokens
    tokens = [
    word for word in tokens
    if word not in stop_words
    ]

    #Stemming
    stemmer=PorterStemmer()

    for index in range(len(tokens)):
        tokens[index]=stemmer.stem(tokens[index])
    
    #back into string(text) from list of token(words)
    text=" ".join(tokens)


    return text


#Reading data from the csv
df=pd.read_csv(r"E:\Customer_Feedback_Sentiment_Analyzer\Data\IMDB_dataset.csv")
print("Before: ",df["review"][0])

#Pandas automatically applies the function to every review.
#.apply(clean_text) takes each value in the review column, 
# sends it into the clean_text() function, and stores the cleaned result back.
df["review"] = df["review"].apply(clean_text)  

print("\n\n\n")
print("After: ",df["review"][0])


#copying it into new file
df.to_csv( r"E:\Customer_Feedback_Sentiment_Analyzer\Data\cleaned_imdb_dataset.csv",index=False)

#index=False prevents Pandas from adding extra row numbers column.