"""
Lemmatization

uses:
    dictionary lookup
    linguistic rules
    sometimes POS tag

Unlike stemming:
    lemmatization uses vocabulary/dictionary knowledge,
    tries to return REAL words.
    Lemmatization is smarter but slower.


| Word    | Stem   | Lemma |
| ------- | ------ | ----- |
| studies | studi  | study |
| running | run    | run   |
| better  | better | good  |

"""
"""
import nltk
nltk.download('wordnet')
nltk.download('punkt_tab')
"""
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
words = ["studies", "playing", "better", "running"]

for word in words:
    print(word, "->", lemmatizer.lemmatize(word,pos='v'))
    """
    Part of Speech (POS)
    | POS   | Meaning   |
    | ----- | --------- |
    | `'n'` | noun      |
    | `'v'` | verb      |
    | `'a'` | adjective |
    | `'r'` | adverb    |

    """
"""
running -> running

instead of: run

Why? 
    Because by default:lemmatizer assumes words are nouns.
    This is a VERY important NLP detail.
    A noun is a word that names a person, place, thing, animal, or idea. Nouns act as the building blocks of sentences and often function as the subject or object.
"""






sentence = "He is running fast"

words = word_tokenize(sentence)

print(pos_tag(words))
"""
| Tag | Meaning     |
| --- | ----------- |
| VBG | verb gerund |
| NN  | noun        |
| JJ  | adjective   |
| RB  | adverb      |

"""


print()
print("Stemming VS lemmatize\n")
words = ["studies", "running", "mice", "better"]

for word in words:

    print("Original:", word)

    print("Stemmed:",
          stemmer.stem(word))

    print("Lemmatized:",
          lemmatizer.lemmatize(word))

    print()
"""
| Aspect                     | Stemming                | Lemmatization                     |
| -------------------------- | ----------------------- | --------------------------------- |
| Main idea                  | Cut words using rules   | Convert to actual dictionary form |
| Logic                      | Character chopping      | Linguistic understanding          |
| Speed                      | Very fast               | Slower                            |
| Accuracy                   | Lower                   | Higher                            |
| Output quality             | Sometimes invalid words | Real meaningful words             |
| Needs dictionary knowledge | ❌                       | ✅                                 |
| Uses grammar/POS           | ❌                       | ✅ (optionally)                    |
| Vocabulary reduction       | Aggressive              | Smarter                           |
| Memory/CPU usage           | Lower                   | Higher                            |
| Best for                   | Classical ML            | Semantic NLP                      |


| Use Case                    | Preferred                |
| --------------------------- | ------------------------ |
| Spam classifier             | Stemming                 |
| Sentiment analysis baseline | Stemming / Lemma both    |
| Search engines              | Often stemming           |
| Chatbots                    | Lemmatization            |
| Semantic search             | Lemmatization            |
| LLM preprocessing           | Usually neither directly |
| Classical SVM/TF-IDF        | Often stemming           |
| Production NLP APIs         | Lemmatization            |

"""
