# Read in data, clean it, and then split into train and test sets
import gensim
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 100)

messages = pd.read_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/spam.csv',
                       encoding='latin-1')
messages = messages.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
messages.columns = ["label", "text"]
messages['text_clean'] = messages['text'].apply(lambda x: gensim.utils.simple_preprocess(x))

X_train, X_test, y_train, y_test = train_test_split(messages['text_clean'],
                                                    messages['label'], test_size=0.2)

# Create tagged document objects to prepare to train the model
tagged_docs = [gensim.models.doc2vec.TaggedDocument(v, [i]) for i, v in enumerate(X_train)]

# Look at what a tagged document looks like
print(tagged_docs[0])

# Train a basic doc2vec model
d2v_model = gensim.models.Doc2Vec(tagged_docs,
                                  vector_size=100,
                                  window=5,
                                  min_count=2)

# What happens if we pass in a single word like we did for word2vec?
d2v_model.infer_vector(['text'])

# What happens if we pass in a list of words?
d2v_model.infer_vector(['i', 'am', 'learning', 'nlp'])

"""### What About Pre-trained Document Vectors?

There are not as many options as there are for word vectors. There also is not an easy API to read these in like there is for `word2vec` so it is more time consuming.

Pre-trained vectors from training on Wikipedia and Associated Press News can be found [here](https://github.com/jhlau/doc2vec). Feel free to explore on your own!
"""