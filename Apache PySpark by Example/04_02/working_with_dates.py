from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate() 

"""## Downloading and preprocessing Chicago's Reported Crime Data"""
from pyspark.sql.functions import to_timestamp, col, lit
rc = spark.read.csv('/Users/yuawong/Downloads/reported-crimes.csv',
                    header=True).withColumn('Date',
                                            to_timestamp(col('Date'),
                                                         'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') <= lit('2018-11-11'))
rc.show(5)

"""## Working with dates"""
from pyspark.sql.functions import to_timestamp, to_date, lit
df = spark.createDataFrame([('2019-12-25 13:30:00',)], ['Christmas'])
df.show(1)
"""  **2019-12-25 13:30:00**"""
df.select(to_date(col('Christmas'), 'yyyy-MM-dd HH:mm:ss'), to_timestamp(col('Christmas'), 'yyyy-MM-dd HH:mm:ss')).show(1)

"""**25/Dec/2019 13:30:00**"""
df = spark.createDataFrame([('25/Dec/2019 13:30:00',)], ['Christmas'])
df.select(to_date(col('Christmas'), 'dd/MMM/yyyy HH:mm:ss'), to_timestamp(col('Christmas'), 'dd/MMM/yyyy HH:mm:ss')).show(1)

"""**12/25/2019 01:30:00 PM**"""
df = spark.createDataFrame([('12/25/2019 01:30:00 PM',)], ['Christmas'])
df.show(1, truncate=False)
df.select(to_date(col('Christmas'), 'MM/dd/yyyy hh:mm:ss aa'),
          to_timestamp(col('Christmas'), 'MM/dd/yyyy hh:mm:ss aa')).show(1)


nrc = spark.read.csv('/Users/yuawong/Downloads/reported-crimes.csv',
                     header=True).withColumn('Date',
                                             to_timestamp(col('Date'),
                                                          'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') <= lit('2018-11-11'))
nrc.show(5)


