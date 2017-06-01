# find spark path
import findspark
findspark.init()

# import necessary packages&methods
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import HashingTF, Tokenizer, Word2Vec
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType
from pyspark.sql.functions import collect_list, concat_ws

# initiate sparksession
spark = SparkSession.builder.appName("test").getOrCreate()

path = "/Users/siqiwang/AnacondaProjects/data/"
# load data with specified StructType
schema_train = StructType([StructField('id', IntegerType(), True),StructField('product_uid1', IntegerType(), True),StructField('product_title', StringType(), True),StructField('search_term', StringType(), True),StructField('relevance', IntegerType(), True)])
traindata = spark.read.csv(path + "train.csv", header=True, mode="DROPMALFORMED", schema=schema_train)
traindata = traindata.select("product_uid1", "product_title", "search_term", "relevance")
# for products that have multiple attributes, group all attributes
schema_attr = StructType([StructField('product_uid2', IntegerType(), True), StructField('name', StringType(), True), StructField('value', StringType(), True)])
attrdata = spark.read.csv(path + "attributes.csv", header=True, mode="DROPMALFORMED", schema=schema_attr)
attrdata = attrdata.select("product_uid2", "value").groupBy("product_uid2").agg(concat_ws(',',collect_list("value")).alias("value_list"))

schema_desc = StructType([StructField('product_uid3', IntegerType(), True), StructField('product_description', StringType(), True)])
descrdata = spark.read.csv(path + "product_descriptions.csv", header=True, mode="DROPMALFORMED", schema=schema_desc)

# show sample lines of data
traindata.printSchema()
traindata.select("*").limit(5).show()
attrdata.printSchema()
attrdata.select("*").limit(5).show()
descrdata.printSchema()
descrdata.select("*").limit(5).show()

# join dataframes
train = traindata.join(attrdata, col("product_uid2") == col("product_uid1"), 'left_outer').join(descrdata, col("product_uid3") == col("product_uid1"), 'left_outer').select("product_uid1", "search_term", "product_title", "value_list", "product_description", "relevance").orderBy("product_uid1")
train.limit(5).show()
train.printSchema()
train.write.option("header", 'true').csv(path + 'saved_file.csv')