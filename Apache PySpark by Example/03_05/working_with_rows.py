import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate() 
spark

"""## Downloading and preprocessing Chicago's Reported Crime Data"""
from pyspark.sql.functions import to_timestamp, col, lit
rc = spark.read.csv('reported-crimes.csv', header=True).withColumn('Date', to_timestamp(col('Date'), 'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') <= lit('2018-11-11'))
rc.show(5)

"""## Working with rows

**Add the reported crimes for an additional day, 12-Nov-2018, to our dataset.**
"""













"""**What are the top 10 number of reported crimes by Primary type, in descending order of occurence?**"""













