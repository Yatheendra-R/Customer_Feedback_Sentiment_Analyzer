import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

"""
Lemmatizer=WordNetLemmatizer()
words = ["running", "fast"]

list_pos_tag=nltk.pos_tag(words)
print(list_pos_tag)

map_pos_tag={"VBG": "v","NN": "n","JJ": "a","RB": "r"}
#pos_tag may be more than this so mapping is practically not possible hence use string or char matching
for index in range(len(words)):
    print(words[index],"->",Lemmatizer.lemmatize(words[index],pos=map_pos_tag[list_pos_tag[index][1]]))
"""
"""
for i in words:
    print(i ," -> ",Lemmatizer.lemmatize(i,pos='v'))
print(wordnet.VERB)
"""
"""
| WordNet POS  | Meaning   |
| ------------ | --------- |
| wordnet.NOUN | noun      |
| wordnet.VERB | verb      |
| wordnet.ADJ  | adjective |
| wordnet.ADV  | adverb    |

| NLTK Tag | WordNet POS |
| -------- | ----------- |
| VBG      | v           |
| NN       | n           |
| JJ       | a           |
| RB       | r           |

WordNet lemmatizer does NOT accept:VBG,RB..   directly.
It only accepts: n , v , a , r

"""




# LEMMATIZER OBJECT
lemmatizer = WordNetLemmatizer()


# SAMPLE WORDS
words = [
    "running",
    "cars",
    "better",
    "studies",
    "quickly"
]


# POS TAGGING


list_pos_tag = nltk.pos_tag(words)

print("POS Tags:\n")
print(list_pos_tag)


# FUNCTION:
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

# POS-AWARE LEMMATIZATION

print("\nLemmatization:\n")

for word, pos_tag in list_pos_tag:

    # CONVERT TAG FORMAT
    wordnet_pos = get_wordnet_pos(pos_tag)

    # LEMMATIZE
    lemma = lemmatizer.lemmatize(
        word,
        pos=wordnet_pos
    )

    # OUTPUT
    print(
        word,
        " | POS:",
        pos_tag,
        " | WordNet POS:",
        wordnet_pos,
        " | Lemma:",
        lemma
    )
"""
Tradeoff Table
| Approach             | Advantage                  | Disadvantage          |
| -------------------- | -------------------------- | --------------------- |
| POS before stopwords | Better grammatical context | Slower                |
| POS after stopwords  | Faster                     | Slightly less context |

"""