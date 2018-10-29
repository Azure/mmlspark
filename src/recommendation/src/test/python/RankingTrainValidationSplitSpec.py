# Prepare training and test data.
import os
import pyspark
import unittest
import xmlrunner
from mmlspark.RankingEvaluator import RankingEvaluator
from mmlspark.RankingTrainValidationSplit import RankingTrainValidationSplit, RankingTrainValidationSplitModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import *
from pyspark.ml.tuning import *
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS


class TrainValidRecommendSplitSpec(unittest.TestCase):

    @staticmethod
    def getRatings():
        cSchema = StructType([StructField("originalCustomerID", IntegerType()),
                              StructField("newCategoryID", IntegerType()),
                              StructField("rating", IntegerType()),
                              StructField("notTime", IntegerType())])

        ratings = pyspark.sql.SparkSession.builder.getOrCreate().createDataFrame([
            (0, 1, 4, 4),
            (0, 3, 1, 1),
            (0, 4, 5, 5),
            (0, 5, 3, 3),
            (0, 7, 3, 3),
            (0, 9, 3, 3),
            (0, 10, 3, 3),
            (1, 1, 4, 4),
            (1, 2, 5, 5),
            (1, 3, 1, 1),
            (1, 6, 4, 4),
            (1, 7, 5, 5),
            (1, 8, 1, 1),
            (1, 10, 3, 3),
            (2, 1, 4, 4),
            (2, 2, 1, 1),
            (2, 3, 1, 1),
            (2, 4, 5, 5),
            (2, 5, 3, 3),
            (2, 6, 4, 4),
            (2, 8, 1, 1),
            (2, 9, 5, 5),
            (2, 10, 3, 3),
            (3, 2, 5, 5),
            (3, 3, 1, 1),
            (3, 4, 5, 5),
            (3, 5, 3, 3),
            (3, 6, 4, 4),
            (3, 7, 5, 5),
            (3, 8, 1, 1),
            (3, 9, 5, 5),
            (3, 10, 3, 3)], cSchema)
        return ratings

    def test_all_tiny(self):
        pyspark.sql.SparkSession.builder.master("local[*]") \
            .config('spark.driver.extraClassPath',
                    "/home/dciborow/mmlspark2/BuildArtifacts/packages/m2/com/microsoft/ml/spark/mmlspark_2.11/0.0/mmlspark_2.11-0.0.jar") \
            .getOrCreate()

        ratings = self.getRatings()

        customerIndex = StringIndexer() \
            .setInputCol("originalCustomerID") \
            .setOutputCol("customerID")

        ratingsIndex = StringIndexer() \
            .setInputCol("newCategoryID") \
            .setOutputCol("itemID")

        pipeline = Pipeline(stages=[customerIndex, ratingsIndex])

        transformedDf = pipeline.fit(ratings).transform(ratings)

        als = ALS() \
            .setUserCol(customerIndex.getOutputCol()) \
            .setRatingCol('rating') \
            .setItemCol(ratingsIndex.getOutputCol())

        paramGrid = ParamGridBuilder() \
            .addGrid(als.regParam, [1.0]) \
            .build()

        evaluator = RankingEvaluator()

        rankingTrainValidationSplit = RankingTrainValidationSplit() \
            .setEstimatorParamMaps(paramGrid) \
            .setEstimator(als) \
            .setEvaluator(evaluator) \
            .setTrainRatio(0.8) \
            .setCollectSubMetrics(True)

        model = rankingTrainValidationSplit.fit(transformedDf)
        print(model.recommendForAllUsers(3))
        print(model.getMetrics())
        print(model.subMetrics())


if __name__ == "__main__":
    result = unittest.main(testRunner=xmlrunner.XMLTestRunner(output=os.getenv("TEST_RESULTS", "TestResults")), \
                           failfast=False, buffer=False, catchbreak=False)