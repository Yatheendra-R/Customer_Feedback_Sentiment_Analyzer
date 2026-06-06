import string
from nltk.corpus import stopwords

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
for i in important_words:
    if i in list_stopwords_english:
        list_stopwords_english.remove(i)

"""
Stopwords contains:
no , not , never , extremely , but ...n.
is VERY important for sentiment. Blindly removing it changes the sentiment of the sentence.
So we should customize the Stopwords to our need 

In sentiment analysis: context-changing words are important.
Especially: negation, intensity, contrast words.

"""
print(list_stopwords_english)
print("Number of Stopwords in english: ",len(list_stopwords_english))
print("Type: ",type(list_stopwords_english))
Review=input("Enter the comment/Review: ")
punctuation_chars=string.punctuation
Review=Review.lower()
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
for short_form, full_form in contractions.items():
    Review = Review.replace(short_form, full_form)
for i in punctuation_chars:
    each_word=Review.split(i)
    Review=" ".join(each_word)

each_word=Review.split()

list_correct_word=[] #after removing stopwords
for i in each_word:
    if(i not in list_stopwords_english):
        list_correct_word.append(i)
Review=" ".join(list_correct_word)
print(Review)


