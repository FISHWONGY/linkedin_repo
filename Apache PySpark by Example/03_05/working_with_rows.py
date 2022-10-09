from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

"""## Downloading and preprocessing Chicago's Reported Crime Data"""
from pyspark.sql.functions import to_timestamp, col, lit
rc = spark.read.csv('/Users/yuawong/Downloads/reported-crimes.csv',
                    header=True).withColumn('Date',
                                            to_timestamp(col('Date'),
                                                         'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') <= lit('2018-11-11'))
rc.show(5)

"""## Working with rows

**Add the reported crimes for an additional day, 12-Nov-2018, to our dataset.**
"""
one_day = spark.read.csv('/Users/yuawong/Downloads/reported-crimes.csv',
                    header=True).withColumn('Date',
                                            to_timestamp(col('Date'),
                                                         'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') == lit('2018-11-12'))

one_day.count()

rc.union(one_day).orderBy('Date', ascending=False).show(5)

"""**What are the top 10 number of reported crimes by Primary type, in descending order of occurence?**"""
rc.groupby('Primary Type').count().orderBy('count', ascending=False).show(10)












