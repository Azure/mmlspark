# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import unittest
from typing import Dict, Set, Union
from pyspark.sql import functions as f
from pyspark.ml import Pipeline
from mmlspark.cyber.feature import *
from mmlspark.cyber.anomaly import AccessAnomaly, AccessAnomalyModel
from mmlsparktest.cyber.dataset import AccessAnomalyConfig, DataFactory
from mmlsparktest.spark import *

epsilon = 10 ** -2


class BasicStats:
    def __init__(self, count: int, min: float, max: float, mean: float, std: float):
        self.count = count
        self.min = min
        self.max = max
        self.mean = mean
        self.std = std

    def as_dict(self):
        return {
            'count': self.count,
            'min': self.min,
            'max': self.max,
            'mean': self.mean,
            'std': self.std
        }

    def __repr__(self):
        return str(self.as_dict())


class StatsMap:
    def __init__(self, stats_map: Dict[str, BasicStats]):
        self.stats_map = stats_map

    def get_stat(self, tenant):
        return self.stats_map.get(tenant)

    def get_tenants(self) -> Set[str]:
        return self.stats_map.keys()

    def get_stats(self) -> Set[BasicStats]:
        return self.stats_map.values()

    def __repr__(self):
        return self.stats_map.__repr__()


def create_stats(df, tenant_col: str, value_col: str = AccessAnomalyConfig.default_output_col) -> BasicStats:
    stat_rows = df.groupBy(tenant_col).agg(
        f.count('*').alias('__count__'),
        f.min(f.col(value_col)).alias('__min__'),
        f.max(f.col(value_col)).alias('__max__'),
        f.mean(f.col(value_col)).alias('__mean__'),
        f.stddev_pop(f.col(value_col)).alias('__std__')
    ).collect()

    stats_map = {row[tenant_col]: BasicStats(
        row['__count__'],
        row['__min__'],
        row['__max__'],
        row['__mean__'],
        row['__std__']
    ) for row in stat_rows}

    return StatsMap(stats_map)


class Dataset:
    def __init__(self):
        num_tenants = 2

        self.training = None
        self.intra_test = None
        self.inter_test = None

        for tid in range(num_tenants):
            training_pdf = DataFactory().create_clustered_training_data(0.25)
            intra_test_pdf = DataFactory().create_clustered_intra_test_data(training_pdf)
            inter_test_pdf = DataFactory().create_clustered_inter_test_data()

            curr_training = sc.createDataFrame(training_pdf).withColumn(
                AccessAnomalyConfig.default_tenant_col, f.lit(tid)
            )

            curr_intra_test = sc.createDataFrame(intra_test_pdf).withColumn(
                AccessAnomalyConfig.default_tenant_col, f.lit(tid)
            )

            curr_inter_test = sc.createDataFrame(inter_test_pdf).withColumn(
                AccessAnomalyConfig.default_tenant_col, f.lit(tid)
            )

            self.training = self.training.union(curr_training) if self.training is not None else curr_training
            self.intra_test = self.intra_test.union(curr_intra_test) if self.intra_test is not None else curr_intra_test
            self.inter_test = self.inter_test.union(curr_inter_test) if self.inter_test is not None else curr_inter_test

        self.training = self.training.cache()
        self.intra_test = self.intra_test.cache()
        self.inter_test = self.inter_test.cache()

        assert self.training.join(
            self.intra_test,
            [AccessAnomalyConfig.default_tenant_col,
             AccessAnomalyConfig.default_user_col,
             AccessAnomalyConfig.default_res_col]
        ).count() == 0

        self.num_users = self.training.select(
            AccessAnomalyConfig.default_tenant_col, AccessAnomalyConfig.default_user_col
        ).distinct().count()

        self.num_resources = self.training.select(
            AccessAnomalyConfig.default_tenant_col, AccessAnomalyConfig.default_res_col
        ).distinct().count()

        self.default_access_anomaly_model = None
        self.per_res_adjust_access_anomaly_model = None
        self.implicit_access_anomaly_model = None

    @staticmethod
    def create_new_training(ratio: float) -> DataFrame:
        training_pdf = DataFactory().create_clustered_training_data(ratio)

        return sc.createDataFrame(training_pdf).withColumn(
            AccessAnomalyConfig.default_tenant_col, f.lit(0)
        ).cache()

    def get_default_access_anomaly_model(self):
        if self.default_access_anomaly_model is not None:
            return self.default_access_anomaly_model

        access_anomaly = AccessAnomalyConfig.create_access_anomaly(max_iter=10)
        self.default_access_anomaly_model = access_anomaly.fit(self.training)
        return self.default_access_anomaly_model


data_set = Dataset()


def get_department(the_col: Union[str, f.Column]) -> f.Column:
    _the_col = the_col if isinstance(the_col, f.Column) else f.col(the_col)
    return f.element_at(f.split(_the_col, '_'), 1)


class TestModelNormalizeTransformer(unittest.TestCase):
    def test_model_standard_scaling(self):
        tenant_col = AccessAnomalyConfig.default_tenant_col
        user_col = AccessAnomalyConfig.default_user_col
        user_vec_col = AccessAnomalyConfig.default_user_col + '_vec'
        res_col = AccessAnomalyConfig.default_res_col
        res_vec_col = AccessAnomalyConfig.default_res_col + '_vec'
        likelihood_col = AccessAnomalyConfig.default_likelihood_col

        df = sc.createDataFrame([
            ['0', 'roy', 'res1', 4.0],
            ['0', 'roy', 'res2', 8.0]],
            [tenant_col, user_col, res_col, likelihood_col]).cache()

        user_model_df = sc.createDataFrame([
            ['0', 'roy', [1.0, 1.0, 0.0, 1.0]]],
            [tenant_col, user_col, user_vec_col]).cache()

        res_model_df = sc.createDataFrame([
            ['0', 'res1', [2.0, 2.0, 1.0, 0.0]],
            ['0', 'res2', [4.0, 4.0, 1.0, 0.0]]],
            [tenant_col, res_col, res_vec_col]).cache()

        user_res_cf_df_model = UserResourceCfDataframeModel(
            tenant_col, user_col, user_vec_col, res_col, res_vec_col, user_model_df, res_model_df
        )

        assert user_res_cf_df_model.check()

        model_normalizer = ModelNormalizeTransformer(df, 2)
        fixed_user_res_cf_df_model = model_normalizer.transform(user_res_cf_df_model)

        assert fixed_user_res_cf_df_model.check()
        assert fixed_user_res_cf_df_model.user_model_df.count() == user_model_df.count()
        assert fixed_user_res_cf_df_model.res_model_df.count() == res_model_df.count()

        assert fixed_user_res_cf_df_model.user_model_df.filter(
            f.size(f.col(user_vec_col)) == 4
        ).count() == user_model_df.count()

        assert fixed_user_res_cf_df_model.res_model_df.filter(
            f.size(f.col(res_vec_col)) == 4
        ).count() == res_model_df.count()

        user_vectors = [row[user_vec_col] for row in fixed_user_res_cf_df_model.user_model_df.collect()]
        assert len(user_vectors) == 1
        assert user_vectors[0] == [-0.5, -0.5, 3.0, -0.5]

    def test_model_end2end(self):
        num_users = 10
        num_resources = 25

        tenant_col = AccessAnomalyConfig.default_tenant_col
        user_col = AccessAnomalyConfig.default_user_col
        user_vec_col = AccessAnomalyConfig.default_user_col + '_vec'
        res_col = AccessAnomalyConfig.default_res_col
        res_vec_col = AccessAnomalyConfig.default_res_col + '_vec'
        likelihood_col = AccessAnomalyConfig.default_likelihood_col

        user_model_df = sc.createDataFrame([
            ('0',
             'roy_{0}'.format(i),
             [float(i % 10), float((i + 1) % (num_users / 2)), 0.0, 1.0]) for i in range(num_users)],
            [tenant_col, user_col, user_vec_col]).cache()

        res_model_df = sc.createDataFrame([
            ('0',
             'res_{0}'.format(i),
             [float(i % 10), float((i + 1) % num_resources / 2), 1.0, 0.0]) for i in range(num_resources)],
            [tenant_col, res_col, res_vec_col]).cache()

        df = (user_model_df
              .select(tenant_col, user_col)
              .distinct()
              .join(res_model_df.select(tenant_col, res_col).distinct(),
                    tenant_col)
              .withColumn(likelihood_col, f.lit(0.0)))

        assert df.count() == num_users * num_resources

        user_res_cf_df_model = UserResourceCfDataframeModel(
            tenant_col, user_col, user_vec_col, res_col, res_vec_col, user_model_df, res_model_df
        )

        assert user_res_cf_df_model.check()

        model_normalizer = ModelNormalizeTransformer(df, 2)

        user_res_norm_cf_df_model = model_normalizer.transform(user_res_cf_df_model)

        assert user_res_cf_df_model.check()
        assert user_res_norm_cf_df_model.user_model_df.count() == user_model_df.count()
        assert user_res_norm_cf_df_model.res_model_df.count() == res_model_df.count()

        assert (user_res_norm_cf_df_model.user_model_df
                .filter(f.size(f.col(user_vec_col)) == 4).count() == user_model_df.count())

        assert (user_res_norm_cf_df_model.res_model_df
                .filter(f.size(f.col(res_vec_col)) == 4).count() == res_model_df.count())

        an_model = AccessAnomalyModel(user_res_norm_cf_df_model, likelihood_col)

        fixed_df = an_model.transform(df)
        assert fixed_df is not None
        assert fixed_df.count() == df.count()

        stats_map = create_stats(fixed_df, tenant_col, likelihood_col)

        for stats in stats_map.get_stats():
            assert stats.min < -epsilon, stats
            assert stats.max > epsilon, stats
            assert abs(stats.mean) < epsilon, stats
            assert abs(stats.std - 1.0) < epsilon, stats


class TestAccessAnomaly(unittest.TestCase):
    def test_enrich_and_normalize(self):
        training = Dataset.create_new_training(1.0).cache()
        access_anomaly = AccessAnomalyConfig.create_access_anomaly(
            max_iter=10
        )

        access_anomaly.setImplicitCF(False)
        access_anomaly.setNegSamplingScore(1.0)
        access_anomaly.setNegSamplingFraction(2.0)

        tenant_col = access_anomaly.getTenantCol()
        user_col = access_anomaly.getUserCol()
        indexed_user_col = 'indexed' + user_col[0:1].upper() + user_col[1:]
        res_col = access_anomaly.getResCol()
        indexed_res_col = 'indexed' + res_col[0:1].upper() + res_col[1:]
        scaled_likelihood_col = access_anomaly.getAccessCol()

        assert training.filter(f.col(user_col).isNull()).count() == 0
        assert training.filter(f.col(res_col).isNull()).count() == 0

        indexer = Pipeline(stages=[
            PartitionedStringIndexer(inputCol=user_col,  partitionKey=tenant_col,
                                     outputCol=indexed_user_col, overlappingIndicies=False),
            PartitionedStringIndexer(inputCol=res_col,  partitionKey=tenant_col,
                                     outputCol=indexed_res_col, overlappingIndicies=False),
        ])

        indexer_model = indexer.fit(training)
        indexed_df = indexer_model.transform(training).cache()

        assert indexed_df.filter(f.col(indexed_user_col).isNull()).count() == 0
        assert indexed_df.filter(f.col(indexed_res_col).isNull()).count() == 0
        assert indexed_df.filter(f.col(indexed_user_col) <= 0).count() == 0
        assert indexed_df.filter(f.col(indexed_res_col) <= 0).count() == 0

        unindexed_df = indexer_model.undo_transform(indexed_df).cache()
        assert unindexed_df.filter(f.col(user_col).isNull()).count() == 0
        assert unindexed_df.filter(f.col(res_col).isNull()).count() == 0

        enriched_indexed_df = access_anomaly._enrich_and_normalize(indexed_df)
        enriched_df = indexer_model.undo_transform(enriched_indexed_df).cache()
        assert enriched_df.filter(f.col(user_col).isNull()).count() == 0
        assert enriched_df.filter(f.col(res_col).isNull()).count() == 0

        assert enriched_df.filter(
            (get_department(user_col) == get_department(res_col)) & (f.col(scaled_likelihood_col) == 1.0)
        ).count() == 0

        assert enriched_df.filter(
            (get_department(user_col) != get_department(res_col)) & (f.col(scaled_likelihood_col) != 1.0)
        ).count() == 0

        assert enriched_df.filter(
            (get_department(user_col) != get_department(res_col))
        ).count() == enriched_df.filter(f.col(scaled_likelihood_col) == 1.0).count()

        assert enriched_df.filter(
            (get_department(user_col) == get_department(res_col))
        ).count() == enriched_df.filter(f.col(scaled_likelihood_col) != 1.0).count()

        low_value = access_anomaly.low_value
        high_value = access_anomaly.high_value

        assert enriched_df.count() > training.count()
        assert enriched_df.filter(
            ((f.col(scaled_likelihood_col) >= low_value) & (f.col(
                scaled_likelihood_col
            ) <= high_value)) | (f.col(scaled_likelihood_col) == 1.0)
        ).count() == enriched_df.count()

    def test_mean_and_std(self):
        model = data_set.get_default_access_anomaly_model()

        res_df = model.transform(data_set.training).cache()
        assert res_df is not None
        assert data_set.training.count() == res_df.count()
        assert res_df.filter(f.col(AccessAnomalyConfig.default_output_col).isNull()).count() == 0

        stats_map = create_stats(res_df, AccessAnomalyConfig.default_tenant_col)

        for stats in stats_map.get_stats():
            assert stats.min < -epsilon, stats
            assert stats.max > epsilon, stats
            assert abs(stats.mean) < epsilon, stats
            assert abs(stats.std - 1.0) < epsilon, stats

    def test_data_match_for_cf(self):
        tenant_col = AccessAnomalyConfig.default_tenant_col
        user_col = AccessAnomalyConfig.default_user_col
        res_col = AccessAnomalyConfig.default_res_col

        df1 = data_set.training.select(
            f.col(tenant_col).alias('df1_tenant'),
            f.col(user_col).alias('df1_user'),
            f.col(res_col).alias('df1_res'),
        )

        df2 = data_set.training.select(
            f.col(tenant_col).alias('df2_tenant'),
            f.col(user_col).alias('df2_user'),
            f.col(res_col).alias('df2_res'),
        )

        df_joined_same_department = (
            df1.join(df2,
                     (df1.df1_tenant == df2.df2_tenant) & (df1.df1_user != df2.df2_user) &
                     (get_department(df1.df1_user) == get_department(df2.df2_user)) &
                     (df1.df1_res == df2.df2_res))
                .groupBy('df1_tenant', df1.df1_user, df2.df2_user)
                .agg(f.count('*').alias('count'))).cache()

        stats_same_map = create_stats(df_joined_same_department, 'df1_tenant', 'count')

        for stats_same in stats_same_map.get_stats():
            assert stats_same.count > 1
            assert stats_same.min == 1
            assert stats_same.max >= 10
            assert stats_same.mean >= 5

        df_joined_diff_department = (
            df1.join(df2,
                     (df1.df1_tenant == df2.df2_tenant) &
                     (df1.df1_user != df2.df2_user) &
                     (get_department(df1.df1_user) != get_department(df2.df2_user)) &
                     (df1.df1_res == df2.df2_res))
                .groupBy('df1_tenant', df1.df1_user, df2.df2_user)
                .agg(f.count('*').alias('count')))

        assert df_joined_diff_department.count() == 0

        assert data_set.intra_test.filter(
            get_department(user_col) == get_department(res_col)
        ).count() == data_set.intra_test.count()

        assert data_set.inter_test.filter(
            get_department(user_col) != get_department(res_col)
        ).count() == data_set.inter_test.count()

    def report_cross_access(self, model: AccessAnomalyModel):
        training_scores = model.transform(data_set.training)
        training_stats: StatsMap = create_stats(training_scores, AccessAnomalyConfig.default_tenant_col)

        print('training_stats')

        for stats in training_stats.get_stats():
            print(stats)
            assert abs(stats.mean) < epsilon, stats.mean
            assert abs(stats.std - 1.0) < epsilon, stats.std

        intra_test_scores = model.transform(data_set.intra_test)
        intra_test_stats = create_stats(intra_test_scores, AccessAnomalyConfig.default_tenant_col)

        inter_test_scores = model.transform(data_set.inter_test)
        inter_test_stats = create_stats(inter_test_scores, AccessAnomalyConfig.default_tenant_col)

        print('test_stats')

        for tid in inter_test_stats.get_tenants():
            intra_stats = intra_test_stats.get_stat(tid)
            inter_stats = inter_test_stats.get_stat(tid)

            print(tid)
            print(intra_stats)
            print(inter_stats)

            assert inter_stats.mean > intra_stats.mean, inter_stats.mean
            assert inter_stats.mean - intra_stats.mean >= 2.0, inter_stats.mean - intra_stats.mean

    def test_cross_access(self):
        self.report_cross_access(data_set.get_default_access_anomaly_model())
