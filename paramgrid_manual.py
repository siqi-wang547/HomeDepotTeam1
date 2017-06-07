# find spark path
import findspark
findspark.init()

# import necessary packages&methods
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("test").getOrCreate()

from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType

# read files
path = "D:\\CMU\\Tasks\\Task15_Big_Data_Analysis\\home_depot_data\\"
# load train data with specified StructType
schema_train = StructType([StructField('id', IntegerType(), True),StructField('product_uid1', IntegerType(), True),StructField('product_title', StringType(), True),StructField('search_term', StringType(), True),StructField('relevance', FloatType(), True)])
traindata = spark.read.csv(path + "train.csv", header=True, mode="DROPMALFORMED", schema=schema_train)
traindata = traindata.select("product_uid1", "product_title", "search_term", "relevance")

# load test data 
schema_test = StructType([StructField('id', IntegerType(), True),StructField('product_uid1', IntegerType(), True),StructField('product_title', StringType(), True),StructField('search_term', StringType(), True)])
testdata = spark.read.csv(path + "test.csv", header=True, mode="DROPMALFORMED", schema=schema_test)
testdata = testdata.select("id", "product_uid1", "product_title", "search_term")

# load product description data
schema_desc = StructType([StructField('product_uid2', IntegerType(), True), StructField('product_description', StringType(), True)])
descrdata = spark.read.csv(path + "product_descriptions.csv", header=True, schema=schema_desc).orderBy("product_uid2")

traindata.show(10)
testdata.show(10)
descrdata.show(10)

from pyspark.sql.functions import regexp_replace, col, when

# join with description
train = traindata.join(descrdata, col("product_uid2") == col("product_uid1"), 'left').drop("product_uid2") \
    .withColumn("product_description", when(col("product_description").isNull(), "empty").otherwise(col("product_description")))

# remove special characters in all fields
train = train.withColumn("product_description", regexp_replace('product_description', '[^a-zA-Z1-9\\s]', '')) \
    .withColumn("product_title", regexp_replace('product_title', '[^a-zA-Z1-9\\s]', '')) \
    .withColumn("search_term", regexp_replace('search_term', '[^a-zA-Z1-9\\s]', ''))

train.orderBy("product_uid1").show(10)

# with reference to: http://www.nltk.org/howto/stem.html
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def stem_tokenizer(sentence):
    return [stemmer.stem(word) for word in sentence.lower().split()]

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType

# tokenize columns with stemmer
tokenizer_udf = udf(stem_tokenizer, ArrayType(StringType()))

tokenized_train = train.withColumn("tokenized_term", tokenizer_udf("search_term")).drop("search_term") \
    .withColumn("tokenized_title", tokenizer_udf("product_title")).drop("product_title") \
    .withColumn("tokenized_desc", tokenizer_udf("product_description")).drop("product_description")

tokenized_train.orderBy("product_uid1").show(10)

# get the length of search terms
def get_search_term_length(tokenized_term):
    return len(tokenized_term)

term_len_udf = udf(get_search_term_length, IntegerType())
train = tokenized_train.withColumn("term_len", term_len_udf("tokenized_term"))

train.show(10)

# get the term frequency in both title and description
def get_freq(terms, words):
    sentence = ''.join(words)
    return sum(int(sentence.find(term) >= 0) for term in terms)

get_freq_udf = udf(get_freq, IntegerType())
train = train.withColumn("tf_title", get_freq_udf("tokenized_term", "tokenized_title")).drop("tokenized_title") \
    .withColumn("tf_desc", get_freq_udf("tokenized_term", "tokenized_desc")).drop("tokenized_desc")

train.show(10)

# join all numbers as feature vector
import numpy as np
from pyspark.ml.linalg import Vectors, DenseVector, VectorUDT

# train.show(5)
def join_vec(term_len, tf_title, tf_desc):
    return Vectors.dense([int(term_len), int(tf_title), int(tf_desc)])

join_vec_udf = udf(join_vec, VectorUDT())
train_feat = train.withColumn("features", join_vec_udf("term_len", "tf_title", "tf_desc")) \
    .drop("tokenized_term", "term_len", "tf_title", "tf_desc")
    
train_feat.show()

# train rf model
from pyspark.ml.regression import RandomForestRegressor 
from pyspark.ml import Pipeline 
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


# random forest estimator
rf = RandomForestRegressor(featuresCol="features",labelCol="relevance", maxDepth=5)

 
# paramgrid, can add param of transformer using addGrid()
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees,[5]) \
    .build()
    

# cross validation
# https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html
from pyspark.ml.evaluation import RegressionEvaluator
cv = TrainValidationSplit(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(labelCol="relevance"),
                          trainRatio=0.8)  
# Run cross-validation, and choose the best set of parameters.
model = cv.fit(train_feat)



# rf = RandomForestRegressor(featuresCol="features",labelCol="relevance", numTrees=15, maxDepth=6)
# model = rf.fit(train_feat)

# prepare the test data following the same steps
test = testdata.join(descrdata, col("product_uid2") == col("product_uid1"), 'left').drop("product_uid2") \
    .withColumn("product_description", when(col("product_description").isNull(), "empty").otherwise(col("product_description")))

# remove special characters in all fields
test = test.withColumn("product_description", regexp_replace('product_description', '[^a-zA-Z1-9\\s]', '')) \
    .withColumn("product_title", regexp_replace('product_title', '[^a-zA-Z1-9\\s]', '')) \
    .withColumn("search_term", regexp_replace('search_term', '[^a-zA-Z1-9\\s]', ''))

test = test.withColumn("tokenized_term", tokenizer_udf("search_term")).drop("search_term") \
    .withColumn("tokenized_title", tokenizer_udf("product_title")).drop("product_title") \
    .withColumn("tokenized_desc", tokenizer_udf("product_description")).drop("product_description")
    
test = test.withColumn("term_len", term_len_udf("tokenized_term")) \
    .withColumn("tf_title", get_freq_udf("tokenized_term", "tokenized_title")).drop("tokenized_title") \
    .withColumn("tf_desc", get_freq_udf("tokenized_term", "tokenized_desc")).drop("tokenized_desc")

test = test.withColumn("features", join_vec_udf("term_len", "tf_title", "tf_desc")) \
    .drop("tokenized_term", "term_len", "tf_title", "tf_desc")
    
test.show(10)

# get prediction from rf model on test

predict = model.transform(test).withColumn("relevance", col("prediction")).drop("product_uid1", "features", "prediction").orderBy('id')

predict.show()
# print predict.count()

# evaluator RMSE
'''
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(labelCol="relevance", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predict)
print("RMSE Root Mean Squared Error = %g" % rmse)
'''
predict.toPandas().to_csv(path + 'paramgrid.csv', index=False)