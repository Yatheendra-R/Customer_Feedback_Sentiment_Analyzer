"""
Stopword Removal

Some words are too common and carry little sentiment meaning.
Examples:is , the , a , an , this , that
These are called: Stopwords
"""
"""
import nltk
nltk.download('stopwords')
import nltk  #NLTK datasets
print(nltk.data.path)
"""
import string
from nltk.corpus import stopwords
list_stopwords_english=stopwords.words('english')
print(list_stopwords_english)
print("Number of Stopwords in english: ",len(list_stopwords_english))
print("Type: ",type(list_stopwords_english))
Review=input("Enter the comment/Review: ")
Punt=string.punctuation
Review=Review.lower()
for i in Punt:
    each_word=Review.split(i)
    Review=" ".join(each_word)

each_word=Review.split()
"""
#Note: Removing is risky and not recommended and professional
# store the correct word in the array  
list_word_toBe_removed=[]
for i in each_word:
    if(i in list_stopwords_english):
        list_word_toBe_removed.append(i)
for i in list_word_toBe_removed:
    each_word.remove(i)"""
list_correct_word=[] #after removing stopwords
for i in each_word:
    if(i not in list_stopwords_english):
        list_correct_word.append(i)
Review=" ".join(list_correct_word)
print(Review)


