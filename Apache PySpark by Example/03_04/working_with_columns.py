import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate() 
spark

"""## Downloading and preprocessing Chicago's Reported Crime Data"""

from pyspark.sql.functions import to_timestamp, col, lit
rc = spark.read.csv('reported-crimes.csv', header=True).withColumn('Date', to_timestamp(col('Date'), 'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') <= lit('2018-11-11'))
rc.show(5)

"""## Working with columns

**Display only the first 5 rows of the column name IUCR **
"""



"""  **Display only the first 4 rows of the column names Case Number, Date and Arrest**"""









"""** Add a column with name One, with entries all 1s **"""







"""** Remove the column IUCR **"""



