from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local').getOrCreate()

pp_df = spark.read.csv("./linkedin_repo/Spark for Machine Learning & AI/data/power_plant.csv",
                       inferSchema=True, header=True)

vectorAssembler = VectorAssembler(inputCols=['AT', 'V', 'AP', 'RH'],
                                  outputCol="features")

# Create a vectorised df
vpp_df = vectorAssembler.transform(pp_df)
vpp_df.take(1)

splits = vpp_df.randomSplit([0.7, 0.3],)
train_df = splits[0]
test_df = splits[1]

# To verify
print(train_df.count())
print(test_df.count())
print(vpp_df.count())

# Create Decision tree object
dt = DecisionTreeRegressor(featuresCol="features", labelCol="PE")
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)

dt_evaluator = RegressionEvaluator(labelCol="PE", predictionCol="prediction", metricName="rmse")

rmse = dt_evaluator.evaluate(dt_predictions)
print(rmse)
# 4.513642521448589

"""
Model 3
Gradient-boost tree regression - Better performance generally, but takes more time to train
"""
from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(featuresCol="features", labelCol="PE")
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)

# Actually, can use the previous evaluator - dt_evaluator
gbt_evaluator = RegressionEvaluator(labelCol="PE", predictionCol="prediction", metricName="rmse")

gbt_rmse = gbt_evaluator.evaluate(gbt_predictions)
print(gbt_rmse) # 3.965391105149242