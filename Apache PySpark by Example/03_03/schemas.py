from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

"""## Downloading and preprocessing Chicago's Reported Crime Data"""

from pyspark.sql.functions import to_timestamp, col, lit
rc = spark.read.csv('/Users/yuawong/Downloads/reported-crimes.csv',
                    header=True).withColumn('Date',
                                            to_timestamp(col('Date'),
                                                         'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') <= lit('2018-11-11'))
rc.show(5)
rc.take(1)
"""## Schemas"""
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, BooleanType, DoubleType, IntegerType
schema = StructType([
    StructField('ID', StringType(), True),
    StructField('Case Number', StringType(), True),
    StructField('Date', TimestampType(), True),
    StructField('Block', StringType(), True),
    StructField('IUCR', StringType(), True),
    StructField('Primary Type', StringType(), True),
    StructField('Description', StringType(), True),
    StructField('Location Description', StringType(), True),
    StructField('Arrest', StringType(), True),
    StructField('Domestic', BooleanType(), True),
    StructField('Beat', StringType(), True),
    StructField('District', StringType(), True),
    StructField('Ward', StringType(), True),
    StructField('Community Area', StringType(), True),
    StructField('FBI Code', StringType(), True),
    StructField('X Coordinate', StringType(), True),
    StructField('Y Coordinate', StringType(), True),
    StructField('Year', IntegerType(), True),
    StructField('Updated On', StringType(), True),
    StructField('Latitude', DoubleType(), True),
    StructField('Longitude', DoubleType(), True),
    StructField('Location', StringType(), True)
])

rc = spark.read.csv('/Users/yuawong/Downloads/reported-crimes.csv', schema=schema)
rc.show(5)







