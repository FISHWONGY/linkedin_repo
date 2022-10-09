# Read in and view the raw data
import pandas as pd

messages = pd.read_csv('./linkedin_repo/Advanced NLP with Python for Machine Learning/data/spam.csv',
                       encoding='latin-1')
messages.head()

# Drop unused columns and label columns that will be used
messages = messages.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "text"]
messages.head()

# How big is this dataset?
messages.shape

# What portion of our text messages are actually spam?
messages['label'].value_counts()

# Are we missing any data?
print('Number of nulls in label: {}'.format(messages['label'].isnull().sum()))
print('Number of nulls in text: {}'.format(messages['text'].isnull().sum()))

