# access anomalies: [complement_access.py](../src/main/python/mmlspark/cyber/anomaly/complement_access.py)
- [Talk at European Spark Conference 2019](https://databricks.com/session_eu19/cybermltoolkit-anomaly-detection-as-a-scalable-generic-service-over-apache-spark)
- [(Internal Microsoft) Talk at MLADS November 2018](https://resnet.microsoft.com/video/42395)
- [(Internal Microsoft) Talk at MLADS June 2019](https://resnet.microsoft.com/video/43618)

1. [ComplementAccessTransformer](../src/main/python/mmlspark/cyber/anomaly/complement_access.py)
   is a SparkML [Transformer](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/ml/Transformer.html).
   Given a dataframe it returns a new dataframe containing new access patterns sampled from
   the set of possible access patterns which did not occur in the given dataframe
   (i.e., it returns a sample from the complement set).

# feature engineering: [indexers.py](../src/main/python/mmlspark/cyber/feature/indexers.py)
1. [IdIndexer](../src/main/python/mmlspark/cyber/feature/indexers.py)
   is a SparkML [Estimator](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/ml/Estimator.html).
   Given a dataframe, it creates an IdIndexerModel (described next) for categorical features which
   contains the information to map each partition and column seen in the given dataframe to an id.
   for each partition or one consecutive range for all partition and column values.
2. [IdIndexerModel](../src/main/python/mmlspark/cyber/feature/indexers.py)
   is a SparkML [Transformer](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/ml/Transformer.html).
   Given a dataframe maps each partition and column field to a consecutive integer id.
   Partitions or column values not encountered in the estimator are mapped to 0.
   The model can operate in two modes, either create consecutive integer id independently 
3. [MultiIndexer](../src/main/python/mmlspark/cyber/feature/indexers.py)
   is a SparkML [Estimator](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/ml/Estimator.html).
   Uses multiple IdIndexer to generate a MultiIndexerModel (described next) for categorical features which
   contains multiple IdIndexers for multiple partitions and columns.
4. [MultiIndexerModel](../src/main/python/mmlspark/cyber/feature/indexers.py)
   is a SparkML [Transformer](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/ml/Transformer.html).
   Given a dataframe maps each partition and column field to a consecutive integer id.
   Partitions or column values not encountered in the estimator are mapped to 0.
   The model can operate in two modes, either create consecutive integer id independently 

# feature engineering: [scalers.py](../src/main/python/mmlspark/cyber/feature/scalers.py)
1. [StandardScalarScaler](../src/main/python/mmlspark/cyber/feature/scalers.py)
   is a SparkML [Estimator](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/ml/Estimator.html).
   Given a dataframe it creates a StandardScalarScalerModel (described next) which normalizes
   any given dataframe according to the mean and standard deviation calculated on the 
   dataframe given to the estimator.
2. [StandardScalarScalerModel](../src/main/python/mmlspark/cyber/feature/scalers.py)
   is a SparkML [Transformer](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/ml/Transformer.html).
   Given a dataframe with a value column x, the transformer changes its value as follows:
   x'=(x-mean)/stddev, i.e., if the transformer is given the same dataframe the estimator 
   was given then the value column will have a mean of 0.0 and a standard deviation of 1.0.
3. [MinMaxScalarScaler](../src/main/python/mmlspark/cyber/feature/scalers.py)
   is a SparkML [Estimator](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/ml/Estimator.html).
   Given a dataframe it creates a MinMaxScalarScalerModel (described next) which normalizes
   any given dataframe according to the minimum and maximum values calculated on the 
   dataframe given to the estimator.
4. [MinMaxScalarScalerModel](../src/main/python/mmlspark/cyber/feature/scalers.py)
   is a SparkML [Transformer](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/ml/Transformer.html).
   Given a dataframe with a value column x, the transformer changes its value such that 
   if the transformer is given the same dataframe the estimator 
   was given then the value column will be scaled linearly to the given ranges.

# access anomalies: [collaborative_filtering.py](../src/main/python/mmlspark/cyber/anomaly/collaborative_filtering.py)
1. [AccessAnomaly](../src/main/python/mmlspark/cyber/anomaly/collaborative_filtering.py)
   is a SparkML [Estimator](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/ml/Estimator.html).
   Given a dataframe the estimator generates an AccessAnomalyModel (next described) which
   can detect anomalous access of users to resources in such a way where the access
   is outside of the user's or resources's profile. For instance a user from HR accessing
   a resource from Finance. This is based solely on access patterns rather than explicit features.
   Internally this is based on Collaborative Filtering as implemented in Spark using 
   Matrix Factorization with Alternating Least Squares.
2. [AccessAnomalyModel](../src/main/python/mmlspark/cyber/anomaly/collaborative_filtering.py)
   is a SparkML [Transformer](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/ml/Transformer.html).
   Given a dataframe the transformer computes a value between (-inf, inf) where positive 
   values indicate an anomaly score. Anomaly scores are computed to have a mean of 1.0
   and a standard deviation of 1.0 over the original dataframe given to the estimator.
3. [ModelNormalizeTransformer](../src/main/python/mmlspark/cyber/anomaly/collaborative_filtering.py)
   is a SparkML [Transformer](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/ml/Transformer.html).
   This is a transformer used internally by AccessAnomaly to normalize a model to generate
   anomaly scores with mean 0.0 and standard deviation of 1.0.
4. [AccessAnomalyConfig](../src/main/python/mmlspark/cyber/anomaly/collaborative_filtering.py)
   contains the default values for AccessAnomaly.
5. [UserResourceCfDataframeModel](../src/main/python/mmlspark/cyber/anomaly/collaborative_filtering.py)
   holds the user and resource dataframes used by the AccessAnomalyModel along 
   with the required column names.
6. [CfAlgoParams](../src/main/python/mmlspark/cyber/anomaly/collaborative_filtering.py)
:   holds parameters for AccessAnomaly which are related to using implicit vs.
   explicit versions of collaborative filtering.
