from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql import SparkSession
import pandas as pd
from sklearn.datasets import load_iris

spark = SparkSession.builder.master('local').getOrCreate()

"""# Get iris dataset
iris = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                   '09_Machine Learning for Time Series Data in Python/data/iris.csv', index_col=0)"""

iris = load_iris()
iris = pd.concat([pd.DataFrame(iris['data']), pd.DataFrame(iris['target'])], axis=1)
iris.columns = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "target"]
# pandas df to spark df
iris_df = spark.createDataFrame(iris)

iris_df.show()
iris_df.take(1)

# Rename col name
iris_df = iris_df.select(col("sepal length (cm)").alias("sepal_length"),
                         col("sepal width (cm)").alias("sepal_width"),
                         col("petal length (cm)").alias("petal_length"),
                         col("petal width (cm)").alias("petal_width"),
                         col("target").alias("species"))

iris_df.take(1)

vectorAssembler = VectorAssembler(inputCols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                                  outputCol="features")

# Create new df
viris_df = vectorAssembler.transform(iris_df)
viris_df.take(1)

indexer = StringIndexer(inputCol="species", outputCol="label")

# to store info from indexer
iviris_df = indexer.fit(viris_df).transform(viris_df)
iviris_df.show(1)


"""
Model 1
0403-Naive Bayes Classification
"""
print(iviris_df)

iviris_df.take(1)

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

splits = iviris_df.randomSplit([0.6, 0.4], 1)
train_df = splits[0]
test_df = splits[1]

# To verify
print(train_df.count())
print(test_df.count())
print(iviris_df.count())

# Build model - Create Naive Bayes Classifier
nb = NaiveBayes(modelType='multinomial')

# Fit model - Create NB model by fitting train dataset to it
nbmodel = nb.fit(train_df)

# Make predictions usinng the model
predictions_df = nbmodel.transform(test_df)
predictions_df.show()

# Look at the df in pandas
predictions_pd = predictions_df.toPandas()

evluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
nbaccuracy = evluator.evaluate(predictions_df)
print(nbaccuracy)
# 0.9807692307692307 iris from sklearn
# 0.7837837837837838 iris from csv
"""
Model 2
0404_Multilayer perceptron classification
"""
from pyspark.ml.classification import MultilayerPerceptronClassifier
layers = [4, 5, 5, 3]

# Create multilayer  perceptron
mlp = MultilayerPerceptronClassifier(layers=layers, seed=1)

mlp_model = mlp.fit(train_df)
mlp_predictions = mlp_model.transform(test_df)

mlp_evluator = MulticlassClassificationEvaluator(metricName="accuracy")

mlp_accuracy = mlp_evluator.evaluate(mlp_predictions)
print(mlp_accuracy)
# 0.9423076923076923

"""
Model 3
0405_Decision tree classification
"""
from pyspark.ml.classification import DecisionTreeClassifier
# label = index version of speices, features = vectors of features
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)

dt_evluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

dt_accuracy = dt_evluator.evaluate(dt_predictions)
print(dt_accuracy)
# 0.9423076923076923 iris from sklearn
# 0.918918918918919 iris from csv