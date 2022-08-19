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


# Standardise numeric data
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
features_df = spark.createDataFrame([
    (1, Vectors.dense([10.0, 10000.0, 1.0]),),
    (2, Vectors.dense([20.0, 30000.0, 2.0]),),
    (3, Vectors.dense([30.0, 40000.0, 3.0]),)
], ["id", "features"])

# Create standard scaler object
feature_stand_scaler = StandardScaler(inputCol="features", outputCol="sfeatures", withStd=True, withMean=True)
# Create model using the scaler object
stand_smodel = feature_stand_scaler.fit(features_df)
stand_sfeatures_df = stand_smodel.transform(features_df)
# check df
stand_sfeatures_df.take(1)
# now all features are standardised to -1 - 1
stand_sfeatures_df.show()
features_df.show()


# Buketise numeric data
from pyspark.ml.feature import Bucketizer
splits = [-float("inf"), -10.0, 0.0, 10.0, float("inf")]

b_data = [(-800.0,), (-10.5,), (-1.7,), (0.0,), (8.2,), (90.1,)]
b_df = spark.createDataFrame(b_data, ["features"])
b_df.show()

# Create bucketiser obj
buckertizer = Bucketizer(splits=splits, inputCol="features", outputCol="bfeatures")
bucketed_df = buckertizer.transform(b_df)
bucketed_df.show()

