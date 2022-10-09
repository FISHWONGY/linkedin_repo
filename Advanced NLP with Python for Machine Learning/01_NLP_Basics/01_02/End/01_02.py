# Import the package and download the necessary data
import nltk
nltk.download()

# What methods and attributes are available in nltk?
dir(nltk)

"""### What Can You Do With NLTK?"""

# Did it work?
from nltk.corpus import stopwords

print(stopwords.words('english')[0:5])

# Let's look at additional words later in the list
print(stopwords.words('english')[0:500:25])

