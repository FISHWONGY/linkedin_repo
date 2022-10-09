import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate() 

from pyspark.sql.functions import to_timestamp,col,lit
rc = spark.read.csv('reported-crimes.csv',header=True).withColumn('Date',to_timestamp(col('Date'),'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') < lit('2018-11-12'))
rc.show(5)

"""## Challenge questions

**What percentage of reported crimes resulted in an arrest?**
"""









"""  **What are the top 3 locations for reported crimes?**"""







