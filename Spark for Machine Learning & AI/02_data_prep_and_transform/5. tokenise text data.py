from pyspark.ml.feature import Tokenizer
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local').getOrCreate()

sentences_df = spark.createDataFrame([(1, "This is an introduction to Spark MLlib"),
                                      (2, "MLlib includes for classifiction and regression"),
                                      (3, "It also contains supporting tools for pipelines")],
                                     ["id", "sentence"])

sentences_df.show()

# Create toekniser object
sent_token = Tokenizer(inputCol="sentence", outputCol="words")
sent_tokenised_df = sent_token.transform(sentences_df)
sent_tokenised_df.show()
sentences_df.take(1)
sent_tokenised_df.take(1)

from pyspark.ml.feature import HashingTF, IDF
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)

# Apply transform
sent_hfTF_df = hashingTF.transform(sent_tokenised_df)

sent_hfTF_df.take(1)

# Scalling depends on words' frequency
idf = IDF(inputCol="rawFeatures", outputCol="idf_features")

# Create model
idfModel = idf.fit(sent_hfTF_df)

# Create df that has both document frequency and transformation frequency
tfidf_df = idfModel.transform(sent_hfTF_df)
tfidf_df.take(1)
