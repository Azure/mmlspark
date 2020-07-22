import unittest

from mmlspark.cyber.utils.spark_utils import ExplainBuilder
from mmlsparktest.spark import *
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params


class TestExplainBuilder(unittest.TestCase):
    class ExplainableObj(Transformer, HasInputCol, HasOutputCol):
        partitionKey = Param(
            Params._dummy(),
            "partitionKey",
            "The name of the column to partition by."
        )

        secondPartitionKey = Param(
            Params._dummy(),
            "secondPartitionKey",
            "The name of the column to partition by."
        )

        def __init__(self):
            super().__init__()
            ExplainBuilder.build(self, inputCol='input', partitionKey=1)
            self.setSecondPartitionKey(2)
            self.setOutputCol('output')

        def _transform(self, dataset):
            pass

    def test_explain(self):
        oo = TestExplainBuilder.ExplainableObj()

        assert oo.getPartitionKey() == 1
        assert oo.partition_key == 1

        assert oo.getSecondPartitionKey() == 2
        assert oo.second_partition_key == 2

        assert oo.getInputCol() == 'input'
        assert oo.input_col == 'input'

        assert oo.getOutputCol() == 'output'
        assert oo.output_col == 'output'

    def test_explain_with_modify(self):
        oo = TestExplainBuilder.ExplainableObj()

        assert oo.partition_key == oo.getPartitionKey()
        assert oo.partition_key == 1

        oo.partition_key = 5

        assert oo.partition_key == 5
        assert oo.getPartitionKey() == 5

    def test_explain_with_modify_with_setter(self):
        oo = TestExplainBuilder.ExplainableObj()

        assert oo.partition_key == oo.getPartitionKey()
        assert oo.partition_key == 1

        oo.setPartitionKey(5)

        assert oo.partition_key == 5
        assert oo.getPartitionKey() == 5


if __name__ == "__main__":
    result = unittest.main()
