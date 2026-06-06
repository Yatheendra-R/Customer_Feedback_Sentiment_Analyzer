"""
import nltk
nltk.download('stopwords')
"""
import string
Punct=string.punctuation 
print("punctuation: "+Punct)
print("Type: ",type(Punct))
Review=input("Enter the Review: ")
Review=Review.lower()
for i in Punct:
    Review_word=Review.split(i)
    Review=" ".join(Review_word) 
print(Review)






