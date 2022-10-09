from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate() 

"""## Downloading and preprocessing Chicago's Reported Crime Data"""
from pyspark.sql.functions import to_timestamp,col,lit
rc = spark.read.csv('/Users/yuawong/Downloads/reported-crimes.csv',
                    header=True).withColumn('Date',
                                            to_timestamp(col('Date'),
                                                         'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') <= lit('2018-11-11'))
rc.show(5)

"""## Working with dates"""











"""  **2019-12-25 13:30:00**"""







"""**25/Dec/2019 13:30:00**"""







"""**12/25/2019 01:30:00 PM**"""





