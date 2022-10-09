import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

"""## Downloading and preprocessing Chicago's Reported Crime Data"""

from pyspark.sql.functions import to_timestamp,col,lit
rc = spark.read.csv('reported-crimes.csv',header=True).withColumn('Date',to_timestamp(col('Date'),'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') <= lit('2018-11-11'))
rc.show(5)

"""## Challenge questions

**What is the most frequently reported non-criminal activity?**
"""

















"""**Using a bar chart, plot which day of the week has the most number of reported crime. 
**
"""















