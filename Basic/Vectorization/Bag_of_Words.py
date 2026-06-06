"""
Vectorization
Machine Learning models cannot understand raw text.

Sentence: "I love this product"
For humans: understandable
For ML model: meaningless text
So we must convert text into: numbers
This process is called: Vectorization
"""

"""
Bag of Words
"""
review_list=["good product","bad service","good quality"]
list_word=[]
set_word=set()
for i in review_list:
    for j in i.split():
        if(j not in set_word):
            list_word.append(j)
            set_word.add(j)

print(list_word)
l=len(list_word)
vector_word=[0]*l
give_sentence="good product"
list_word_sentence=give_sentence.split()

for i in list_word_sentence:
    j=list_word.index(i)
    vector_word[j]=1

print(vector_word)

"""
Problem With Basic Bag of Words=>Common words may dominate.
Example: good good good good
Word frequency alone may not represent importance properly.
"""