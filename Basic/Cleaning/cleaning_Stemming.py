

import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    list_stopwords_english=stopwords.words('english')
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
    #removing important_words from Stopwords
    for i in important_words:
        if i in list_stopwords_english:
            list_stopwords_english.remove(i)

    print(list_stopwords_english)
    print("Number of Stopwords in english: ",len(list_stopwords_english))
    print("Type: ",type(list_stopwords_english))

    punctuation_chars=string.punctuation #geting all punctuation
    
    text=text.lower() #lowering
    
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
    #Remaping contractions
    for short_form, full_form in contractions.items():
        text = text.replace(short_form, full_form)
    
    #Removing punctuation_chars
    for i in punctuation_chars:
        each_word=text.split(i)
        text=" ".join(each_word)

    each_word=text.split()

    list_correct_word=[] #after removing stopwords

    #Omiting stopwords
    for i in each_word:
        if(i not in list_stopwords_english):
            list_correct_word.append(i)
    text=" ".join(list_correct_word)

    return text


text=input("Enter the comment/text: ")
text=preprocess_text(text)
print(text)

