# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys

if sys.version >= "3":
    basestring = str

from pyspark.ml.util import JavaMLReadable, JavaMLReader, MLReadable
from pyspark.ml.wrapper import JavaParams
from pyspark.ml.common import inherit_doc, _java2py
from pyspark import SparkContext

@inherit_doc
class JavaMMLReadable(MLReadable):
    """
    (Private) Mixin for instances that provide JavaMLReader.
    """

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return JavaMMLReader(cls)

@inherit_doc
class ComplexParamsMixin(MLReadable):
    def _transfer_params_from_java(self):
        """
        Transforms the embedded com.microsoft.ml.spark.core.serialize.params from the companion Java object.
        """
        sc = SparkContext._active_spark_context
        for param in self.params:
            if self._java_obj.hasParam(param.name):
                java_param = self._java_obj.getParam(param.name)
                # SPARK-14931: Only check set com.microsoft.ml.spark.core.serialize.params back to avoid default com.microsoft.ml.spark.core.serialize.params mismatch.
                complex_param_class = sc._gateway.jvm.com.microsoft.ml.spark.core.serialize.ComplexParam._java_lang_class
                is_complex_param = complex_param_class.isAssignableFrom(java_param.getClass())
                if self._java_obj.isSet(java_param):
                    if is_complex_param:
                        value = self._java_obj.getOrDefault(java_param)
                    else:
                        value = _java2py(sc, self._java_obj.getOrDefault(java_param))
                    self._set(**{param.name: value})

@inherit_doc
class JavaMMLReader(JavaMLReader):
    """
    (Private) Specialization of :py:class:`MLReader` for :py:class:`JavaParams` types
    """

    def __init__(self, clazz):
        super(JavaMMLReader, self).__init__(clazz)

    @classmethod
    def _java_loader_class(cls, clazz):
        """
        Returns the full class name of the Java ML instance.
        """
        return clazz.getJavaPackage()
