##################################################
# Churn Prediction using PySpark
##################################################
import warnings
import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

findspark.init("C:\spark\spark-3.1.1-bin-hadoop2.7")

spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_giris") \
    .getOrCreate()

sc = spark.sparkContext
# sc.stop()

spark_df = spark.read.csv("datasets/churn2.csv", header=True, inferSchema=True)

##################################################
# EDA
##################################################

# gözlem ve değişken sayısı
print("Shape:", (spark_df.count(), len(spark_df.columns)))

# değişken tipleri
spark_df.printSchema()
spark_df.dtypes

# değişken seçme
spark_df.Age

# head
spark_df.head(5)
spark_df.show(5)

# değişken isimlerinin küçültülmesi
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])
spark_df.show(5)

# özet istatistikler
spark_df.describe().show()

# belirli değişkenler için özet istatistikler
spark_df.describe(["age", "creditscore"]).show()

# Tüm numerik değişkenlerin seçimi ve özet istatistikleri
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
spark_df.select(num_cols).describe().toPandas().transpose()

# Tüm kategorik değişkenlerin seçimi ve özeti
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']

for col in cat_cols:
    spark_df.select(col).distinct().show()

# Churn'e göre sayısal değişkenlerin özet istatistikleri
for col in num_cols:
    spark_df.groupby("exited").agg({col: "mean"}).show()

# filter(): Gözlem seçimi / filtreleme
spark_df.filter(spark_df.age > 40).show()
spark_df.filter(spark_df.age > 40).count()

##################################################
# DATA PREPROCESSING & FEATURE ENGINEERING
##################################################

##################################################
# Missing Values
##################################################
from pyspark.sql.functions import when, count, col

spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T

spark_df.dropna().show()

############################
# Bucketizer ile Değişken Türetmek/Dönüştürmek
############################
from pyspark.ml.feature import Bucketizer

bucketizer = Bucketizer(splits=[0, 35, 45, 100], inputCol="age", outputCol="age_cat")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df = spark_df.withColumn('age_cat', spark_df.age_cat + 1)
spark_df = spark_df.withColumn("age_cat", spark_df["age_cat"].cast("integer"))
spark_df.show(20)
spark_df = spark_df.drop("age")

spark_df = spark_df.withColumn('gender', when(spark_df['gender'] == "Female", "1").otherwise("0").cast("integer"))

spark_df = spark_df.withColumn('Geography_cat',
                               when(spark_df['Geography'] == 'France', 1).
                               when(spark_df['Geography'] == 'Spain', 2).otherwise(3))

spark_df = spark_df.withColumn('tenure_cat',
                               when(spark_df['tenure'] < 3, "new_transaction").
                               when((3 < spark_df['tenure']) & (spark_df['tenure'] < 6), "middle_transaction").
                               otherwise("former_transaction"))

spark_df.show(20)
##################################################
# Label Encoding
##################################################
indexer_geo = StringIndexer(inputCol="Geography_cat", outputCol="Geography_label")
temp_sdf = indexer_geo.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("Geography_label", temp_sdf["Geography_label"].cast("integer"))
spark_df = spark_df.drop('Geography_cat')

indexer_tenure = StringIndexer(inputCol="tenure_cat", outputCol="tenure_label")
temp_sdf = indexer_tenure.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("tenure_label", temp_sdf["tenure_label"].cast("integer"))
spark_df = spark_df.drop('tenure_cat')
##################################################
# One Hot Encoding
##################################################
encoder_age = OneHotEncoder(inputCols=["age_cat"], outputCols=["age_cat_ohe"])
encoder_tenure = OneHotEncoder(inputCols=["tenure_label"], outputCols=["tenure_label_ohe"])

spark_df = encoder_tenure.fit(spark_df).transform(spark_df)
spark_df = encoder_age.fit(spark_df).transform(spark_df)

spark_df = spark_df.drop("tenure_label")
spark_df = spark_df.drop("age_cat")

##################################################
# TARGET'ın Tanımlanması
##################################################
stringIndexer = StringIndexer(inputCol='exited', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))
spark_df.show(10)

##################################################
# FEATURE'LARIN Tanımlanması
##################################################
cols = ['creditscore', 'gender', 'tenure', 'balance', 'numofproducts', 'hascrcard', 'isactivemember', 'estimatedsalary',
        'age_cat_ohe', 'tenure_label_ohe', 'Geography_label']

va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)
va_df.show()

# Final sdf
final_df = va_df.select("features", "label")
final_df.show(5)

train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))

##################################################
# MODELING - Logistic Regression
##################################################

log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_df)
y_pred = log_model.transform(test_df)
y_pred.show()

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')

evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))

##################################################
# GBM
##################################################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)

y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()


##################################################
# MODEL TUNİNG
##################################################

evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)


y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()
