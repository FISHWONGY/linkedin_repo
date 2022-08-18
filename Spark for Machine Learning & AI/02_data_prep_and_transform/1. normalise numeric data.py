from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local').getOrCreate()
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors

features_df = spark.createDataFrame([
    (1, Vectors.dense([10.0, 10000.0, 1.0]),),
    (2, Vectors.dense([20.0, 30000.0, 2.0]),),
    (3, Vectors.dense([30.0, 40000.0, 3.0]),)
], ["id", "features"])

features_df.show()
features_df.take(1)

feature_scaler = MinMaxScaler(inputCol="features", outputCol="sfeatures")
smodel = feature_scaler.fit(features_df)

sfeatures_df = smodel.transform(features_df)
# Look at first row of data
sfeatures_df.take(1)
# Look at the df, now we can see at the new col - sfeatures, the value are all scaled to 0-1
sfeatures_df.select("features", "sfeatures").show()

