# Read in and clean data
import nltk
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import string

stopwords = nltk.corpus.stopwords.words('english')

messages = pd.read_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/spam.csv',
                       encoding='latin-1')
messages = messages.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
messages.columns = ["label", "text"]
messages['label'] = np.where(messages['label'] == 'spam', 1, 0)


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    return text


messages['clean_text'] = messages['text'].apply(lambda x: clean_text(x))
print(messages.head())

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(messages['clean_text'],
                                                    messages['label'], test_size=0.2)

# What do the first ten messages in the training set look like?
print(X_train[:])

# What do the labels look like?
print(y_train[:10])

# Let's save the training and test sets to ensure we are using the same data for each model
X_train.to_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/X_train.csv',
               index=False, header=True)
X_test.to_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/X_test.csv',
              index=False, header=True)
y_train.to_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/y_train.csv',
               index=False, header=True)
y_test.to_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/y_test.csv',
              index=False, header=True)

