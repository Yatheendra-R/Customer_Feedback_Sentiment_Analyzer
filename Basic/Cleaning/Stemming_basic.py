"""
Stemming is a text normalization technique in NLP that reduces words to their base or root form by stripping away prefixes and suffixes.
Its primary goal is to standardize words so that variations of the same word are treated as a single token, improving text analysis and search efficiency.

Stemming uses rule-based heuristics to crudely chop off word endings. It is fast and simple,
but because it relies on fixed rules rather than linguistic understanding, it doesn't always guarantee a real dictionary word.

Example:Original words: connects, connected, connection, connectingStemmed 
output: connect (valid word)Original words: trouble, troubling, troubledStemmed output: troubl (not a valid dictionary word)

Stemming is most widely used in applications where processing speed and broad matching take priority over grammatical perfection:

There are several standard algorithms used to perform stemming:
Porter Stemmer: One of the oldest and most common algorithms. It removes suffixes using a set of cascading rules (e.g., changing -ed or -ing to nothing).
Snowball Stemmer (Porter2): An improved, slightly more aggressive version of the Porter Stemmer. It supports multiple languages.
Lancaster Stemmer: A highly aggressive algorithm that utilizes iterative rules, often reducing words to very short, sometimes unrecognizable roots.

Stemming vs. LemmatizationStemming should not be confused with Lemmatization, which is a more advanced normalization technique.
While stemming simply cuts off word endings, lemmatization uses vocabulary, grammar rules, and part-of-speech (POS)
analysis to return a valid dictionary word (e.g., changing better to good).
"""


"""
Stemming
The goal of stemming is: reduce similar words to a common root form.
playing,played.plays->play
This helps the ML model treat related words similarly.
Without stemming:are treated as DIFFERENT features.
With stemming:they become more unified.
Unifying words through stemming reduces them to their core root. This eliminates minor variations and condenses the total number of unique words (dimensionality) in a dataset.
Improves Search and Recall,Boosts Machine Learning Accuracy,Enables Accurate Sentiment Analysis etc

"""
"""
Step 1:
Remove plurals

Step 2:
Remove tense suffixes

Step 3:
Remove derivational suffixes

Step 4:
Cleanup

relational
↓
relate
↓
relat
"""
from nltk.stem import PorterStemmer
"""
NLTK library
└── stem module
    └── PorterStemmer class
"""
stemmer=PorterStemmer() #This creates a stemmer object.
 #stemming is mostly rule-based suffix reduction.
words = ["playing", "played", "plays", "studies", "happiness"]
for word in words:
    print(word+" -> "+stemmer.stem(word))
#stemmer.stem(word) This calls the stemming algorithm.
# method inside the PorterStemmer class,stemmer is object of PorterStemmer class

"""
Important Limitation

Porter Stemmer:
    does not understand context
    does not know POS tags
    does not know real dictionary roots
"""