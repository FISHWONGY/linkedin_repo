# Load pretrained word vectors using gensim
import gensim.downloader as api

wiki_embeddings = api.load('glove-wiki-gigaword-100')

# Explore the word vector for "king"
wiki_embeddings['king']

# Find the words most similar to king based on the trained word vectors
wiki_embeddings.most_similar('king')

"""### Train Our Own Model"""

# Read in the data and clean up column names
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 100)

messages = pd.read_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/spam.csv', encoding='latin-1')
messages = messages.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
messages.columns = ["label", "text"]
messages.head()

# Clean data using the built in cleaner in gensim
messages['text_clean'] = messages['text'].apply(lambda x: gensim.utils.simple_preprocess(x))
messages.head()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(messages['text_clean'],
                                                    messages['label'], test_size=0.2)

# Train the word2vec model
w2v_model = gensim.models.Word2Vec(X_train,
                                   size=100,
                                   window=5,
                                   min_count=2)

# Explore the word vector for "king" base on our trained model
w2v_model.wv['king']

# Find the most similar words to "king" based on word vectors from our trained model
w2v_model.wv.most_similar('king')

