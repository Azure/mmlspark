from pyspark.sql import functions as f, types as t, SparkSession, SQLContext

def setUpModule():
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("TestIndexers") \
        .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:" + os.environ["MML_VERSION"]) \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.sql.shuffle.partitions", 10) \
        .getOrCreate()

    spark_context = SQLContext(spark.sparkContext)

def tearDownModule():
    pass
