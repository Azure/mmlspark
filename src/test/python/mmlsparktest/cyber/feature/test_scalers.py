# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import unittest
from pyspark.sql import functions as f, types as t
from mmlspark.cyber.feature import *
from mmlsparktest.spark import *

class TestScalers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = sc.createDataFrame(
            [
                ("t1", "5", 500.0),
                ("t1", "6", 600.0),
                ("t2", "7", 700.0),
                ("t2", "8", 800.0),
                ("t3", "9", 900.0)
            ],
            ["tenant", "name", "score"]
        ).cache()

    @classmethod
    def tearDownClass(cls):
        cls.df.unpersist()

    def test_unpartitioned_min_max_scaler(self):
        ls = PartitionedMinMaxScaler(
            inputCol='score',
            outputCol='new_score',
            minValue=5.0,
            maxValue=9.0)

        new_df = ls.fit(self.df).transform(self.df).cache()

        assert new_df.count() == self.df.count()

        assert 0 == new_df.filter(
            f.col('name').cast(t.IntegerType()) != f.col('new_score').cast(t.IntegerType())
        ).count()

    def test_partitioned_min_max_scaler(self):
        ls = PartitionedMinMaxScaler(
            inputCol='score',
            partitionKey='tenant',
            outputCol='new_score',
            minValue=1.0,
            maxValue=2.0)

        new_df = ls.fit(self.df).transform(self.df).cache()

        assert new_df.count() == self.df.count()

        t1_arr = new_df.filter(f.col('tenant') == 't1').orderBy('new_score').collect()
        assert len(t1_arr) == 2
        assert t1_arr[0]['new_score'] == 1.0
        assert t1_arr[1]['new_score'] == 2.0

        t2_arr = new_df.filter(f.col('tenant') == 't2').orderBy('new_score').collect()
        assert len(t2_arr) == 2
        assert t2_arr[0]['new_score'] == 1.0
        assert t2_arr[1]['new_score'] == 2.0

        t3_arr = new_df.filter(f.col('tenant') == 't3').orderBy('new_score').collect()
        assert len(t3_arr) == 1
        # this is the average between min and max
        assert t3_arr[0]['new_score'] == 1.5

    def test_unpartitioned_standard_scaler(self):
        ls = PartitionedStandardScaler(
            inputCol='score',
            outputCol='new_score',
            std=1.0)

        new_df = ls.fit(self.df).transform(self.df).cache()

        assert new_df.count() == self.df.count()

        new_scores = new_df.toPandas()['new_score']

        assert abs(new_scores.to_numpy().mean()) < 0.0001
        assert abs(new_scores.to_numpy().std() - 1.0) < 0.0001

    def test_partitioned_standard_scaler(self):
        ls = PartitionedStandardScaler(
            inputCol='score', partitionKey='tenant', outputCol='new_score', std=1.0)

        new_df =  ls.fit(self.df).transform(self.df).cache()

        assert new_df.count() == self.df.count()

        for tenant in ['t1', 't2', 't3']:
            new_scores = new_df.filter(f.col('tenant') == tenant).toPandas()['new_score']

            assert new_scores is not None

            the_mean = new_scores.to_numpy().mean()
            the_std = new_scores.to_numpy().std()
            tenant_scores = [s for _, s in new_scores.items()]

            assert the_mean == 0.0

            if tenant != 't3':
                assert abs(the_std - 1.0) < 0.0001, str(the_std)
                assert len(tenant_scores) == 2
                assert tenant_scores[0] == -1.0
                assert tenant_scores[1] == 1.0
            else:
                assert the_std == 0.0
                assert len(tenant_scores) == 1
                assert tenant_scores[0] == 0.0


if __name__ == "__main__":
    result = unittest.main()
