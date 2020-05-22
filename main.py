from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import sparknlp

# spark = sparknlp.start()
# Had to start session manuallt rather than with sparknlp.start() to tune memory preferences
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

document = DocumentAssembler()\
    .setInputCol("description")\
    .setOutputCol("document")

# use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
#  .setInputCols(["document"])\
#  .setOutputCol("sentence_embeddings")

use = UniversalSentenceEncoder.load("model/")\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")


# bert = BertEmbeddings.load('/tmp/bert_base_cased_en_2.4.0_2.4_1580579557778')\
# .setInputCols(["sentence",'token'])\
# .setOutputCol("bert")\
# .setCaseSensitive(False)\
# .setPoolingLayer(0)

# classifier_dl = ClassifierDLApproach()\
#   .setInputCols(["sentence_embeddings"])\
#   .setOutputCol("class")\
#   .setLabelColumn("category")\
#   .setMaxEpochs(1)\
#   .setEnableOutputLogs(True)
#
# use_clf_pipeline = Pipeline(
#     stages = [
#         document,
#         use,
#         classifier_dl
#     ])
#
# use_pipelineModel = use_clf_pipeline.fit(trainDataset)

print("\n\n\n#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n\n\n")
