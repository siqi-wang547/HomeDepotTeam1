# find spark path
import findspark
findspark.init()

# import necessary packages&methods
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql.functions import collect_list, concat_ws

# initiate sparksession
spark = SparkSession.builder.appName("test").getOrCreate()

path = "D:\\CMU\\Tasks\\Task15_Big_Data_Analysis\\home_depot_data\\"
# load train data with specified StructType
schema_train = StructType([StructField('id', IntegerType(), True),StructField('product_uid1', IntegerType(), True),StructField('product_title', StringType(), True),StructField('search_term', StringType(), True),StructField('relevance', FloatType(), True)])
traindata = spark.read.csv(path + "train.csv", header=True, mode="DROPMALFORMED", schema=schema_train)
traindata = traindata.select("product_uid1", "product_title", "search_term", "relevance")
# for products that have multiple attributes, group all attributes
schema_attr = StructType([StructField('product_uid2', IntegerType(), True), StructField('name', StringType(), True), StructField('value', StringType(), True)])
attrdata = spark.read.csv(path + "attributes.csv", header=True, schema=schema_attr)
attrdata = attrdata.select("product_uid2", "value").groupBy("product_uid2").agg(concat_ws(' ',collect_list("value")).alias("attr_list")).orderBy("product_uid2")

# load test data 
schema_test = StructType([StructField('id', IntegerType(), True),StructField('product_uid1', IntegerType(), True),StructField('product_title', StringType(), True),StructField('search_term', StringType(), True)])
testdata = spark.read.csv(path + "test.csv", header=True, mode="DROPMALFORMED", schema=schema_test)
testdata = testdata.select("id", "product_uid1", "product_title", "search_term")


# load product description data
schema_desc = StructType([StructField('product_uid3', IntegerType(), True), StructField('product_description', StringType(), True)])
descrdata = spark.read.csv(path + "product_descriptions.csv", header=True, schema=schema_desc).orderBy("product_uid3")

# show sample lines of data
#print traindata.count()
traindata.show()
#print attrdata.count()
attrdata.show()
#print descrdata.count()
descrdata.show()
#print testdata.count()
testdata.show()

from pyspark.sql.functions import concat, lit, col, when

# union train and test
train_and_test = traindata.drop("relevance").union(testdata.drop('id'))

# join dataframes
train = train_and_test.join(attrdata, col("product_uid2") == col("product_uid1"), 'left').drop("product_uid2")
train = train.withColumn("attr_list", when(col("attr_list").isNull(), "empty").otherwise(col("attr_list")))
train = train.join(descrdata, col("product_uid3") == col("product_uid1"), 'left').drop("product_uid3")
train = train.withColumn("product_description", when(col("product_description").isNull(), "empty").otherwise(col("product_description")))
# combine relevant fields (product_title, attr_list, product_description, search_term) as one named `info`
train = train.select("product_uid1", "search_term", concat(col("product_title"), lit(' '), col("attr_list"), lit(' '), col("product_description"), lit(' '), col("search_term")).alias("info")).orderBy("product_uid1")
train.show()
# train.printSchema()
# train.write.option("header", 'true').csv(path + 'saved_file.csv')

from pyspark.ml.feature import StopWordsRemover, Tokenizer, Word2Vec

# tokenize `info` and remove stopwords in unioned data
tokenizer = Tokenizer(inputCol="info", outputCol="tokenized_info")
tokenized = tokenizer.transform(train).drop("info")
tokenized.show()
# remover = StopWordsRemover(inputCol="tokenized_info", outputCol="filtered_info")
# removed = remover.transform(tokenized).drop("tokenized_info").na.drop()
# print removed.count()

# use union tokenized 
word2Vec = Word2Vec(vectorSize=3, minCount = 5, inputCol="tokenized_info", outputCol="vec")
model = word2Vec.fit(tokenized)
# model.getVectors().show()
# tokenized search terms in train.csv
term_tokenizer = Tokenizer(inputCol="search_term", outputCol="term_token")
term_tokenized = term_tokenizer.transform(traindata).drop("search_term")
term_tokenized.show()

# use word2vec to transform search terms in train.csv into feature vectors
trained = model.transform(term_tokenized.withColumn("tokenized_info", col('term_token')))
trained.show()

# tokenize test data and transform search terms into vectors
tokenized_test = term_tokenizer.transform(testdata).drop("search_term")
tokenized_test.show()
test_vector = model.transform(tokenized_test.withColumn("tokenized_info", col('term_token')))

# from pyspark.ml.regression import LinearRegression
# train linear model
# lr = LinearRegression(maxIter=10, featuresCol="vec", regParam=0.0, elasticNetParam=0.8, labelCol="relevance")
# lr_model = lr.fit(trained)
# prediction = lr_model.transform(test_vector)

# random forest model 
# reference: https://spark.apache.org/docs/1.6.2/ml-classification-regression.html
from pyspark.ml.regression import RandomForestRegressor 
from pyspark.ml.feature import StringIndexer, VectorIndexer 
from pyspark.ml import Pipeline 
rf = RandomForestRegressor(featuresCol="features")
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(trained)
labelIndexer = StringIndexer(inputCol="relevance", outputCol="label").fit(trained)
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])
rf_model = pipeline.fit(trained)
prediction = model.transform(test_vector)

# for row in selected.collect():
#     rid, text, prediction = row
#     print("(%d, %s) --> prediction=%f" % (rid, text, prediction))

# print linear regresssion coefficients
# print("Coefficients: %s" % str(lr_model.coefficients))
# print("Intercept: %s" % str(lr_model.intercept))
# check predictions
prediction.select('id', "prediction").show()

# save csv to local
reqdata = prediction.select(col('id'), col("prediction").alias("relevance"))
reqdata.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save(path + 'rf_word2vec')