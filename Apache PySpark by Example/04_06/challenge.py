from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

"""## Downloading and preprocessing Chicago's Reported Crime Data"""
from pyspark.sql.functions import to_timestamp, col, lit
rc = spark.read.csv('/Users/yuawong/Downloads/reported-crimes.csv',
                    header=True).withColumn('Date',
                                            to_timestamp(col('Date'),
                                                         'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') <= lit('2018-11-11'))
rc.show(5)

"""## Challenge questions

**What is the most frequently reported non-criminal activity?**
"""
rc.select(col('Primary Type')).distinct().count()
rc.select(col('Primary Type')).distinct().orderBy('Primary Type', ascending=False).show(36, truncate=False)

nc = rc.filter((col('Primary Type') == "NON-CRIMINAL (SUBJECT SPECIFIED)") |
               (col('Primary Type') == "NON-CRIMINAL") |
               (col('Primary Type') == "NON - CRIMINAL"))
nc.show(5)
nc.groupby('Description').count().orderBy('count', ascending=False).show(3)
"""**Using a bar chart, plot which day of the week has the most number of reported crime. 
**
"""
from pyspark.sql.functions import dayofweek, date_format
rc.select(col('Date'), dayofweek(col('Date'))).show(5)

rc.select(col('Date'), dayofweek(col('Date')), date_format(col('Date'), 'E')).show(5)

rc.groupby(date_format(col('Date'), 'E')).count().orderBy('count', ascending=False).show(7)

dow = [x[0] for x in rc.groupby(date_format(col('Date'), 'E')).count().collect()]
count = [x[1] for x in rc.groupby(date_format(col('Date'), 'E')).count().collect()]







