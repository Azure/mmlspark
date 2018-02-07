# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for information.

import numpy as np


# Methods for train-test split.
class TrainTestSplit:

    @staticmethod
    def min_rating_filter(self, min_rating, by_customer):
        '''
        Filter rating DataFrame for each user with minimum rating.
        :param min_rating: minimum number of rating for filtering.
        :param by: by which variable (customer or item) to filter the rating.
        '''
        from pyspark.sql.functions import col

        if by_customer:
            by = "customer"
            with_ = "item"
            split_by_column = by + "ID"
            split_with_column = "item" + "ID"
        else:
            by = "item"
            with_ = "customer"
            split_by_column = by + "ID"
            split_with_column = "customer" + "ID"

        rating_filtered = self.groupBy(split_by_column) \
            .agg({split_with_column: "count"}) \
            .withColumnRenamed('count(' + split_with_column + ')', "n" + split_with_column) \
            .where(col("n" + split_with_column) >= min_rating) \
            .join(self, split_by_column) \
            .drop("n" + split_with_column)

        return rating_filtered

    @staticmethod
    def stratified_split(self, min_rating, by_customer=True, ratio=0.3, fixed_test_sample=False, sample=3):
        '''
        Perform stratified sampling on rating DataFrame to split into train and test.
        Fixed ratio and fixed number also apply to splitting. The fixed number of samples for testing should be less than min_rating.
        This method is usually used for evaluating ranking metrics for warm user or item.
        :param min_rating: minimum number of rating for filtering.
        :param ratio: splitting ratio for train and test.
        :param by: by which variable (customer or item) to filter the rating.
        '''
        from pyspark.sql import Window
        from pyspark.sql.functions import row_number, col, rand, bround

        if fixed_test_sample == True & sample > min_rating:
            print("sample should be less than min_rating.")
            return -1

        if by_customer:
            by = "customer"
            with_ = "item"
            split_by_column = by + "ID"
            split_with_column = "item" + "ID"
        else:
            by = "item"
            with_ = "customer"
            split_by_column = by + "ID"
            split_with_column = "customer" + "ID"

        rating_joined = self.min_rating_filter(min_rating, by_customer)

        window_spec = Window.partitionBy(split_by_column).orderBy(rand())

        rating_grouped = rating_joined.groupBy(split_by_column) \
            .agg({split_with_column: "count"}) \
            .withColumnRenamed("count(" + split_with_column + ")", "count")

        if fixed_test_sample == False:
            rating_all = rating_joined.join(rating_grouped, on = split_by_column, how="outer") \
                .withColumn('splitPoint', bround(col('count') * ratio))

            rating_train = rating_all \
                .select('*', row_number().over(window_spec).alias('rank')) \
                .filter(col('rank') > col('splitPoint'))
            rating_test = rating_all \
                .select('*', row_number().over(window_spec).alias('rank')) \
                .filter(col('rank') <= col('splitPoint'))
        else:
            rating_train = rating_joined \
                .select('*', row_number().over(window_spec).alias('rank')) \
                .filter(col('rank') > sample)
            rating_test = rating_joined \
                .select('*', row_number().over(window_spec).alias('rank')) \
                .filter(col('rank') <= sample)

        return rating_train, rating_test

    @staticmethod
    def chronological_split(self, min_rating, by_customer=True, ratio=0.3, fixed_test_sample=False, sample=3):
        '''
        Chronological splitting split data (items are ordered by timestamps for each customer) by timestamps.
        Fixed ratio and fixed number also apply to splitting. The fixed number of samples for testing should be less than min_rating.
        This method assumes implicit rating so there must be timestamps presented in the DataFrame.
        This method is usually used for evaluating ranking metrics for warm user or item.
        :param min_rating: minimum number of rating for filtering.
        :param ratio: sampling ratio for testing set .
        :param fixed_test_sample: whether or not fixing the number in sampling testing data.
        :param sample: number of samples for testing data.
        '''
        from pyspark.sql import Window
        from pyspark.sql.functions import col, bround, row_number

        if fixed_test_sample == True & sample > min_rating:
            print("sample should be less than min_rating.")
            return -1

        if by_customer:
            by = "customer"
            with_ = "item"
            split_by_column = by + "ID"
            split_with_column = "item" + "ID"
        else:
            by = "item"
            with_ = "customer"
            split_by_column = by + "ID"
            split_with_column = "customer" + "ID"

        rating_joined = self.min_rating_filter(min_rating, by_customer)

        rating_grouped = rating_joined \
            .groupBy(split_by_column) \
            .agg({'timeStamp': 'count'}) \
            .withColumnRenamed('count(timeStamp)', 'count')

        window_spec = Window.partitionBy(split_by_column).orderBy(col('timeStamp').desc())

        if fixed_test_sample == False:
            rating_all = rating_joined.join(rating_grouped, on = split_by_column, how="outer") \
                .withColumn('splitPoint', bround(col('count') * ratio))

            rating_train = rating_all \
                .select('*', row_number().over(window_spec).alias('rank')) \
                .filter(col('rank') > col('splitPoint'))
            rating_test = rating_all \
                .select('*', row_number().over(window_spec).alias('rank')) \
                .filter(col('rank') <= col('splitPoint'))
        else:
            rating_train = rating_joined \
                .select('*', row_number().over(window_spec).alias('rank')) \
                .filter(col('rank') > sample)
            rating_test = rating_joined \
                .select('*', row_number().over(window_spec).alias('rank')) \
                .filter(col('rank') <= sample)

        rating_train = rating_train.select(split_by_column, split_with_column, "timeStamp")
        rating_test = rating_test.select(split_by_column, split_with_column, "timeStamp")

        return rating_train, rating_test

    @staticmethod
    def non_overlapping_split(self, min_rating, by_customer=True, ratio=0.7):
        '''
        Split by customer or item. Customer (or item) in sets of training and testing data are mutually exclusive.
        This method is usually used for evaluating ranking metrics for cold user or item.
        :param min_rating: minimum number of rating for filtering.
        :param ratio: sampling ratio for testing set .
        '''
        from pyspark.sql.window import Window
        from pyspark.sql.functions import row_number, col, rand

        rating_joined = self.min_rating_filter(min_rating, by_customer)

        if by_customer:
            by = "customer"
            with_ = "item"
            split_by_column = by + "ID"
            split_with_column = "item" + "ID"
        else:
            by = "item"
            with_ = "customer"
            split_by_column = by + "ID"
            split_with_column = "customer" + "ID"

        rating_exclusive = rating_joined.groupBy(split_by_column) \
            .agg({split_with_column: "count"}) \
            .withColumnRenamed("count(" + split_with_column + ")", "n" + with_) \
            .drop("n" + with_)

        count = rating_exclusive.count()

        window_spec = Window.orderBy(rand())
        rating_tmp = rating_exclusive.select(col("*"), row_number().over(window_spec).alias("rowNumber"))

        rating_split = \
            rating_tmp.filter(rating_tmp['rowNumber'] <= round(count * ratio)).drop("rowNumber"), \
            rating_tmp.filter(rating_tmp['rowNumber'] > round(count * ratio)).drop('rowNumber')

        rating_train = rating_joined.join(rating_split[0], split_by_column)
        rating_test = rating_joined.join(rating_split[1], split_by_column)

        return rating_train, rating_test

    @staticmethod
    def random_split(self, min_rating, by_customer=True, ratio=0.7):
        '''
        Purely random splitting.
        This method is generally used for evaluating rating metrics for both warm and cold user/item.
        :param min_rating: minimum number of rating for filtering.
        :param ratio: sampling ratio for testing set .
        '''
        from pyspark.sql.window import Window
        from pyspark.sql.functions import row_number, col, rand

        rating_split = self.min_rating_filter(min_rating, by_customer) \
            .randomSplit([1 - ratio, ratio])

        rating_train = rating_split[0]
        rating_test = rating_split[1]

        return rating_train, rating_test


if __name__ == "__main__":
    print("Splitter")
