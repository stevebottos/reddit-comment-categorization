from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import sparknlp

spark = SparkSession.builder \
    .appName("Spark NLP")\
    .config("spark.driver.memory","1G")\
    .config("spark.driver.maxResultSize", "1G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.0")\
    .config("spark.kryoserializer.buffer.max", "500M")\
    .getOrCreate()
print("\n\n\n#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n\n\n")

trainDataset = spark.read \
      .option("header", True) \
      .csv("nlp_dat.csv")
trainDataset.show(10, truncate=50)
