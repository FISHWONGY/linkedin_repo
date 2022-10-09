from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

"""## Downloading and preprocessing Chicago's Reported Crime Data"""

from pyspark.sql.functions import to_timestamp, col, lit
rc = spark.read.csv('/Users/yuawong/Downloads/reported-crimes.csv',
                    header=True).withColumn('Date',
                                            to_timestamp(col('Date'),
                                                         'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') <= lit('2018-11-11'))
rc.show(5)

"""## Joins

**Download police station data**
"""
ps = spark.read.csv('/Users/yuawong/Downloads/Police_Stations.csv', header=True)
ps.show(5)

"""**The reported crimes dataset has only the district number. Add the district name by joining with the police station dataset**"""
rc.cache()
rc.count()

ps.select(col('DISTRICT')).distinct().show(30)
rc.select(col('DISTRICT')).distinct().show(30)

from pyspark.sql.functions import lpad
ps.select(lpad(col('DISTRICT'), 3, '0')).show(30)

ps = ps.withColumn('Format_district', lpad(col('DISTRICT'), 3, '0'))
ps.show(5)

ps.columns
rc.join(ps, rc.District == ps.Format_district, how='left_outer').drop('ADDRESS', 'CITY', 'STATE', 'ZIP', 'WEBSITE',
                                                                      'PHONE', 'FAX', 'TTY', 'X COORDINATE', 'Y COORDINATE', 'LATITUDE', 'LONGITUDE', 'LOCATION').show(3)






