"""
Raw Text
↓
Lowercase
↓
Expand contractions
↓
Remove HTML
↓
Remove punctuation/special chars
↓
Normalize spaces
↓
Tokenize
↓
stopword handling
↓
negation propagation
↓
POS tagging
↓
Convert POS tags
↓
POS-aware lemmatization
↓
Join back to text
"""
#Importing 
import string
import re
import pandas as pd
import nltk
from nltk.stem  import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

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

    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is"
}

# NEGATION HANDLING
negation_words = {
    "not",
    "no",
    "never"
}

# LEMMATIZER OBJECT
lemmatizer = WordNetLemmatizer()

# Stopwords
stop_words = set(stopwords.words("english"))

# Important sentiment words to preserve
important_words = {

        "not",
        "no",
        "nor",
        "never",

        "but",
        "very",
        "too",
        "only",

        "more",
        "most",
        "less",

        "against",
        "again",

        "down",
        "up",

        "off",
        "over",
        "under"    
}

# Remove important words from stopwords
stop_words -= important_words


# CONVERT NLTK POS TAG -> WORDNET POS TAG


def get_wordnet_pos(treebank_tag):

    
    # NOUN FAMILY

    if treebank_tag.startswith("N"):
        return wordnet.NOUN

    
    # VERB FAMILY
    
    elif treebank_tag.startswith("V"):
        return wordnet.VERB

    # ADJECTIVE FAMILY
    
    elif treebank_tag.startswith("J"):
        return wordnet.ADJ

    
    # ADVERB FAMILY
    elif treebank_tag.startswith("R"):
        return wordnet.ADV

    # DEFAULT FALLBACK
    else:
        return wordnet.NOUN

# -----------------------------
# NEGATION HANDLING
# -----------------------------
def apply_negation(tokens):
    result = []
    negate = False
    window = 3
    count = 0

    for word in tokens:
        if word in negation_words:
            negate = True
            count = window
            result.append(word)
            continue

        if negate:
            result.append("NOT_" + word)
            count -= 1
            if count == 0:
                negate = False
        else:
            result.append(word)

    return result

def preprocess(text):

    #lower
    text=text.lower()

    for short_form, full_form in contractions.items():
        text = text.replace(short_form, full_form)

    #regx
    #remove HTML tags
    text=re.sub(r"<.*?>"," ",text)

    # REMOVE URLS
    text = re.sub(r"http\S+|www\S+", " ", text)

    # KEEP ONLY LETTERS + SPACES
    text = re.sub(r"[^a-zA-Z_\s]", " ", text)
    
    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    #Tokenize
    tokens=text.split()

    #REMOVE STOPWORDS

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]
    
    # negation handling
    tokens = apply_negation(tokens)


    list_pos_tag = nltk.pos_tag(tokens)

    new_tokens=[]
    for word, pos_tag in list_pos_tag:

        # CONVERT TAG FORMAT
        wordnet_pos = get_wordnet_pos(pos_tag)

        # LEMMATIZE
        lemma = lemmatizer.lemmatize(
            word,
            pos=wordnet_pos
        )
        new_tokens.append(lemma)
    tokens=new_tokens

    return " ".join(tokens)
        




    




df=pd.read_csv(r"E:\Customer_Feedback_Sentiment_Analyzer\Data\IMDB_dataset.csv")

print(df.columns)
X=df[df.columns[0]] #column1
y=df[df.columns[1]] #column1

print("\nBefore Cleaning Sample:\n", df["review"][0])
df["review"] = df["review"].apply(preprocess)
print("\nAfter Cleaning Sample:\n", df["review"][0])


output_path = r"E:\Customer_Feedback_Sentiment_Analyzer\Data\cleaned_imdb_dataset_lemma.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")


