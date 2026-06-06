"""
Core TF-IDF Idea

TF-IDF tries to answer:

“How important is this word in this document compared to all documents?”

Intuition
Words appearing everywhere:
movie
product
service

are less informative.

Rare but meaningful words:
excellent
terrible
amazing
awful

are more useful for sentiment.

So TF-IDF gives:

lower weight to common words,
higher weight to informative words.

Formula Concept

TF-IDF(t,d)=TF(t,d)×log(
DF(t)
N
	​

)


| Part | Meaning                         |
| ---- | ------------------------------- |
| TF   | frequency in current document   |
| DF   | how common across all documents |
| IDF  | rarity importance               |

Important Intuition

If word appears:

in EVERY review → low importance
in FEW reviews → high importance
That intuition is NOT universally correct.
mportant limitation of TF-IDF.
TF-IDF is only a heuristic/statistical assumption.
Not perfect truth.

TF-IDF does NOT “understand language.”

It only uses:

frequency statistics,
distribution patterns.

Machine learning is often:

useful approximation

NOT:

perfect understanding
Then Why Does TF-IDF Still Work Well?

Because across LARGE datasets:

Rare words often carry more distinguishing information statistically.
| Word      | Sentiment Value |
| --------- | --------------- |
| excellent | strong positive |
| horrible  | strong negative |
| delivery  | neutral/common  |
| movie     | generic         |

Over many documents,
this heuristic works surprisingly well.

Compare Human vs TF-IDF
Human Understanding
not good

Human understands:

negation,
context,
semantics.  


TF-IDF Understanding

It only sees weighted tokens like:

not → 0.42
good → 0.67

No true comprehension

This Is Why Modern NLP Exists

Advanced systems like:

Word2Vec,
GloVe,
BERT,
Transformers,
LLMs

try to capture:

context,
meaning,
semantic relationships.

Unlike TF-IDF.

TF-IDF gives WEIGHTS, not simple counts.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

reviews = [
    "good product",
    "bad service",
    "good quality"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(reviews)
#Learn vocabulary + Learn IDF weights + Convert training reviews into vectors
"""
fit() → learns vocabulary + IDF values
transform() → converts text into vectors
| Step              | What Happens                                       |
| ----------------- | -------------------------------------------------- |
| `fit()`           | Learn vocabulary + IDF statistics                  |
| `fit_transform()` | Learn + convert training text into vectors         |
| `transform()`     | Convert new text using existing learned vocabulary |

"""
print(vectorizer.get_feature_names_out())

print(X.toarray())