from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

"""## Downloading and preprocessing Chicago's Reported Crime Data"""

from pyspark.sql.functions import to_timestamp, col, lit
rc = spark.read.csv('/Users/yuawong/Downloads/reported-crimes.csv',
                    header=True).withColumn('Date',
                                            to_timestamp(col('Date'),
                                                         'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') <= lit('2018-11-11'))
rc.show(5)

"""## Working with columns

**Display only the first 5 rows of the column name IUCR **
"""
rc.select('IUCR').show(5)
rc.select(rc.IUCR).show(5)
rc.select(col('IUCR')).show(5)

"""  **Display only the first 4 rows of the column names Case Number, Date and Arrest**"""
rc.select('Case Number', "Date", 'Arrest').show(4)


"""** Add a column with name One, with entries all 1s **"""
from pyspark.sql.functions import lit
rc.withColumn('One', lit(1)).show(5)

"""** Remove the column IUCR **"""
rc = rc.drop('IUCR')
rc.show(5)

