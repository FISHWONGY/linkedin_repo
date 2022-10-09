# Read in raw data and clean up the column names
import pandas as pd
pd.set_option('display.max_colwidth', 100)

messages = pd.read_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/spam.csv',
                       encoding='latin-1')
messages = messages.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
messages.columns = ["label", "text"]
print(messages.head())

"""### Remove Punctuation"""

# What punctuation is included in the default list?
import string

print(string.punctuation)

# Why is it important to remove punctuation?

"This message is spam" == "This message is spam."


# Define a function to remove punctuation in our messages
def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    return text


messages['text_clean'] = messages['text'].apply(lambda x: remove_punct(x))

print(messages.head())

"""### Tokenize"""

# Define a function to split our sentences into a list of words
import re


def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens


messages['text_tokenized'] = messages['text_clean'].apply(lambda x: tokenize(x.lower()))

print(messages.head())

"""### Remove Stopwords"""

# What does an example look like?

tokenize("I am learning NLP".lower())

# Load the list of stopwords built into nltk
import nltk

stopwords = nltk.corpus.stopwords.words('english')


# Define a function to remove all stopwords
def remove_stopwords(tokenized_text):    
    text = [word for word in tokenized_text if word not in stopwords]
    return text


messages['text_nostop'] = messages['text_tokenized'].apply(lambda x: remove_stopwords(x))

print(messages.head())

# Remove stopwords in our example
remove_stopwords(tokenize("I am learning NLP".lower()))

