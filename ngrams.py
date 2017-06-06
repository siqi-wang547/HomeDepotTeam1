# find spark path
import findspark
findspark.init()

# import necessary packages&methods
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.ml.feature import StopWordsRemover, Tokenizer, Word2Vec, NGram
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import concat, lit, col, when

# initiate sparksession
spark = SparkSession.builder.appName("test").getOrCreate()

path = "tmp/"
# load train data with specified StructType
schema_train = StructType([StructField('id', IntegerType(), True),StructField('product_uid1', IntegerType(), True),StructField('product_title', StringType(), True),StructField('search_term', StringType(), True),StructField('relevance', FloatType(), True)])
traindata = spark.read.csv(path + "train.csv", header=True, mode="DROPMALFORMED", schema=schema_train)
traindata = traindata.select("product_uid1", "product_title", "search_term", "relevance")

tokenizer = Tokenizer(inputCol="search_term", outputCol="words")

# 'n' value changed here. Be sure to change saved file name 'NgramSolutionN[n]' to avoid file duplication errors below.
ngram = NGram(n=1, inputCol="words", outputCol="nGrams")
word2Vec = Word2Vec(vectorSize=3, minCount = 5, inputCol="nGrams", outputCol="features")

lr = LinearRegression(maxIter=10, featuresCol="features", regParam=0.0, elasticNetParam=0.8, labelCol="relevance")
pipeline = Pipeline(stages=[tokenizer, ngram, word2Vec, lr])
# Fit the pipeline to training documents.
model = pipeline.fit(traindata)

schema_test = StructType([StructField('id', IntegerType(), True),StructField('product_uid1', IntegerType(), True),StructField('product_title', StringType(), True),StructField('search_term', StringType(), True)])
testdata = spark.read.csv(path + "test.csv", header=True, mode="DROPMALFORMED", schema=schema_test)
testdata = testdata.select("id", "product_uid1", "product_title", "search_term")

rescaledData = model.transform(testdata)

#rescaledData.show()
reqdata = rescaledData.select(col('id'), col("prediction").alias("relevance"))
# Be sure to change the file name 'NgramSolutionN[n]' below to match the n-value above to avoid file duplication errors.
reqdata.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save(path + 'NgramSolutionN1')