from pyspark.ml.regression import LinearRegression
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

# Create Linear regression object
lr = LinearRegression(featuresCol="features", labelCol="PE")
lr_model = lr.fit(vpp_df)

print(lr_model.coefficients) # [-1.9775131067284113,-0.23391642256928327,0.06208294364801217,-0.1580541029343498]
print(lr_model.intercept) # 454.6092744523414
print(lr_model.summary.rootMeanSquaredError) # 4.557126016749488, error around 1%

# to save the model
lr_model.save("./linkedin_repo/Spark for Machine Learning & AI/data/lr1.model")
