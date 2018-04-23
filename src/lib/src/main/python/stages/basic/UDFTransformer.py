# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys
if sys.version >= '3':
    basestring = str

from pyspark.ml.param.shared import *
from mmlspark.stages.basic._UDFTransformer import _UDFTransformer
from pyspark import keyword_only

from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer, JavaEstimator, JavaModel

from pyspark.ml.common import inherit_doc
from pyspark.sql.functions import UserDefinedFunction
from pyspark.ml.common import inherit_doc
from mmlspark.Utils import *

@inherit_doc
class UDFTransformer(ComplexParamsMixin, JavaMLReadable, JavaMLWritable, JavaTransformer):
    """
    Args:

        inputCol (str): The name of the input column (default: )
        outputCol (str): The name of the output column
        udf (object): User Defined Python Function to be applied to the DF input col
        udfScala (object): User Defined Function to be applied to the DF input col
    """

    @keyword_only
    def __init__(self, inputCol=None, inputCols=None, outputCol=None, udf=None):
        super(UDFTransformer, self).__init__()
        self._java_obj = self._new_java_obj("com.microsoft.ml.spark.stages.basic.UDFTransformer")
        self.inputCol = Param(self, "inputCol", "inputCol: The name of the input column (default: )")
        self.inputCols = Param(self, "inputCols", "inputCols: The names of the input columns (default: )")
        self.outputCol = Param(self, "outputCol", "outputCol: The name of the output column")
        self.udf = Param(self, "udf", "udf: User Defined Python Function to be applied to the DF input col")
        if udf != None:
            self.setUDF(udf)
            self._udf = udf
        if inputCol != None:
            self.setInputCol(inputCol)
        if inputCols != None:
            self.setInputCols(inputCols)
        if outputCol != None:
            self.setOutputCol(outputCol)

    def setInputCol(self, value):
        """

        Args:

            inputCol (str): The name of the input column (default: )

        """
        self._set(inputCol=value)
        return self


    def getInputCol(self):
        """

        Returns:

            str: The name of the input column (default: )
        """
        return self.getOrDefault(self.inputCol)


    def setInputCols(self, value):
        """

        Args:

            inputCols (list): The names of the input columns (default: )

        """
        self._set(inputCols=value)
        return self

    def getInputCols(self):
        """

        Returns:

            str: The name of the input column (default: )
        """
        return self.getOrDefault(self.inputCols)

    def setOutputCol(self, value):
        """

        Args:

            outputCol (str): The name of the output column

        """
        self._set(outputCol=value)
        return self


    def getOutputCol(self):
        """

        Returns:

            str: The name of the output column
        """
        return self.getOrDefault(self.outputCol)

    def setUDF(self, udf):
        name = getattr(udf,"_name",  getattr(udf, "__name__", None))
        name = name if name else udf.__class__.__name__
        userDefinedFunction = UserDefinedFunction(udf.func, returnType = udf.returnType, name = name)
        self._java_obj = self._java_obj.setUDF(userDefinedFunction._judf)
        self._udf = udf
        return self

    def getUDF(self):
        return self._udf

    @classmethod
    def read(cls):
        """ Returns an MLReader instance for this class. """
        return JavaMMLReader(cls)

    @staticmethod
    def getJavaPackage():
        """ Returns package name String. """
        return "com.microsoft.ml.spark.stages.basic.UDFTransformer"

    @staticmethod
    def _from_java(java_stage):
        module_name=_UDFTransformer.__module__
        module_name=module_name.rsplit(".", 1)[0] + ".UDFTransformer"
        return from_java(java_stage, module_name)
