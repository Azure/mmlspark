# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import unittest
from pyspark.sql import functions as f
from mmlspark.cyber.feature import *
from mmlsparktest.spark import *
from pyspark.ml import Pipeline


class TestIndexers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = sc.createDataFrame(
            [
                ('1', 'a', 'A', 0, 0),
                ('1', 'b', 'A', 1, 0),
                ('1', 'a', 'B', 0, 1),
                ('2', 'aa', 'AA', 1, 0),
                ('2', 'bb', 'AA', 0, 0),
                ('3', 'b', 'B', 0, 0)
            ],
            ["tenant", "user", "res", "expected_uid", "expected_rid"]
        ).cache()

    @classmethod
    def tearDownClass(cls):
        cls.df.unpersist()

    def test_id_indexer(self):
        indexer = PartitionedStringIndexer(
            inputCol='user', partitionKey='tenant', outputCol='actual_uid', overlappingIndicies=True)
        new_df = indexer.fit(self.df).transform(self.df).cache()
        assert new_df.count() == self.df.count()
        assert 0 == new_df.filter(f.col('expected_uid') != f.col('actual_uid')).count()


    def test_inverse(self):
        indexer = PartitionedStringIndexer(
            inputCol='user', partitionKey='tenant', outputCol='actual_uid', overlappingIndicies=True)
        fit_model = indexer.fit(self.df)
        new_df = fit_model.transform(self.df).withColumnRenamed("user", "user_2")
        new_df_2 = fit_model.inverseTransform(new_df)
        assert 0 == new_df_2.filter(f.col('user') != f.col('user_2')).count()

    def test_multi_indexer(self):
        multi_indexer = Pipeline(stages=[
            PartitionedStringIndexer(inputCol='user', partitionKey='tenant',
                                     outputCol='actual_uid', overlappingIndicies=True),
            PartitionedStringIndexer(inputCol='res', partitionKey='tenant',
                                     outputCol='actual_rid', overlappingIndicies=True),
        ])
        new_df = multi_indexer.fit(self.df).transform(self.df).cache()
        assert new_df.count() == self.df.count()
        assert new_df.filter(f.isnull(f.col('actual_uid'))).count() == 0
        assert new_df.filter(f.isnull(f.col('actual_rid'))).count() == 0
        assert 0 == new_df.filter(f.col('expected_uid') != f.col('actual_uid')).count()
        assert 0 == new_df.filter(f.col('expected_rid') != f.col('actual_rid')).count()


if __name__ == "__main__":
    result = unittest.main()
