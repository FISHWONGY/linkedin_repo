# Load the cleaned training and test sets
import gensim
import numpy as np
import pandas as pd

X_train = pd.read_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/X_train.csv')
X_test = pd.read_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/X_test.csv')
y_train = pd.read_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/y_train.csv')
y_test = pd.read_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/y_test.csv')

"""### Create word2vec Vectors"""

# Train a basic word2vec model
w2v_model = gensim.models.Word2Vec(X_train,
                                   vector_size=100,
                                   window=5,
                                   min_count=2)

# Replace the words in each text message with the learned word vector
words = set(w2v_model.wv.index_to_key)
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_train['clean_text']])
X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_test['clean_text']])

# Average the word vectors for each sentence (and assign a vector of zeros if the model
# did not learn any of the words in the text message during training
X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(100, dtype=float))
        
X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(100, dtype=float))

# What does the unaveraged version look like?
print(X_train_vect[0])

# What does the averaged version look like?
print(X_train_vect_avg[0])

"""### Fit RandomForestClassifier On Top Of Word Vectors"""

# Instantiate and fit a basic Random Forest model on top of the vectors
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf_model = rf.fit(X_train_vect_avg, y_train.values.ravel())

# Use the trained model to make predictions on the test data
y_pred = rf_model.predict(X_test_vect_avg)

# Evaluate the predictions of the model on the holdout test set
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred == y_test['label']).sum()/len(y_pred), 3)))
# Precision: 0.566 / Recall: 0.204 / Accuracy: 0.874
