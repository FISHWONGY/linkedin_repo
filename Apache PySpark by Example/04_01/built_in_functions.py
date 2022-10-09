from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

"""## Downloading and preprocessing Chicago's Reported Crime Data"""

from pyspark.sql.functions import to_timestamp, col, lit
rc = spark.read.csv('/Users/yuawong/Downloads/reported-crimes.csv',
                    header=True).withColumn('Date',
                                            to_timestamp(col('Date'),
                                                         'MM/dd/yyyy hh:mm:ss a')).filter(col('Date') <= lit('2018-11-11'))
rc.show(5)

"""## Built-in functions"""

from pyspark.sql import functions

print(dir(functions))

"""## String functions

**Display the Primary Type column in lower and upper characters, and the first 4 characters of the column**
"""
from pyspark.sql.functions import lower, upper, substring
help(substring)
rc.printSchema()
rc.select(lower(col('Primary Type')), upper(col('Primary Type')), substring(col('Primary Type'), 1, 4)).show(5)

"""## Numeric functions

**Show the oldest date and the most recent date**
"""
from pyspark.sql.functions import min, max
rc.select(min(col('Date')), max(col('Date'))).show(1)

"""##Date

** What is 3 days earlier that the oldest date and 3 days later than the most recent date?**
"""
from pyspark.sql.functions import date_add, date_sub
rc.select(date_sub(min('Date'), 3), date_add(max('Date'), 3)).show(1)






