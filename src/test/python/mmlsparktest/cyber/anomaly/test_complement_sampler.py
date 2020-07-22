# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import unittest

from mmlspark.cyber.anomaly import ComplementSampler
from mmlsparktest.spark import *
from pyspark.sql import functions as f


class TestComplementSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = sc.createDataFrame(
            [
                ("t1", 0, 0),
                ("t1", 1, 1),
                ("t1", 2, 2),
                ("t1", 3, 3),
                ("t1", 4, 4),
                ("t1", 5, 5),
                ("t2", 0, 0),
                ("t2", 1, 1),
                ("t2", 2, 2),
                ("t2", 3, 3)
            ],
            ["tenant", "user", "res"]
        ).cache()

    @classmethod
    def tearDownClass(cls):
        cls.df.unpersist()

    def max_value(self, df, tenant, colname):
        return (df
        .filter(f.col('tenant') == tenant)
        .agg(f.max(colname).alias(colname))
        .first()[colname])

    def test_partitioned_complement_sampler(self):
        complement_df = ComplementSampler(
            partitionKey='tenant', inputCols=['user', 'res'], samplingFactor=3.0) \
            .transform(self.df).cache()

        assert complement_df.count() > 0
        assert complement_df.schema == self.df.schema
        #assert complement_df.select('tenant', 'user', 'res').distinct().count() == complement_df.count()
        assert complement_df.join(self.df, ['tenant', 'user', 'res']).count() == 0
        assert self.max_value(complement_df, "t1", "user") <= 5
        assert self.max_value(complement_df, "t1", "res") <= 5
        assert self.max_value(complement_df, "t2", "user") <= 3
        assert self.max_value(complement_df, "t2", "res") <= 3


    def test_unpartitioned_complement_sampler(self):
        df = self.df.filter(f.col('tenant') == 't1').select('user', 'res').cache()
        assert df is not None and df.count() > 0

        complement_df = ComplementSampler(inputCols=['user', 'res'], samplingFactor=3.0).transform(df).cache()
        assert complement_df.count() > 0
        assert complement_df.schema == df.drop("tenant").schema

        #assert complement_df.select('user', 'res').distinct().count() == complement_df.count()
        assert complement_df.join(df, ['user', 'res']).count() == 0
        assert (df.agg(f.max("user").alias("user"))
            .first()["user"]) <= 5
        assert (df.agg(f.max("res").alias("res"))
            .first()["res"]) <= 5


if __name__ == "__main__":
    result = unittest.main()
