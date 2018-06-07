# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys

if sys.version >= '3':
    basestring = str

from mmlspark._CNTKLearner import _CNTKLearner
from mmlspark.CNTKModel import CNTKModel as CNTKmod
from pyspark.ml.common import inherit_doc
from os.path import join

@inherit_doc
class CNTKLearner(_CNTKLearner):
    """
    Create CNTK model from existing java model

    Args:
        java_model (py4j.java_gateway.JavaObject): see Scala CNTKModel documentation

    """
    def _create_model(self, java_model):
        model = CNTKmod()
        model._java_obj = java_model
        model._transfer_params_from_java()
        return model

    def setPythonModel(self, value, learnerCode):
        """

        Args:
            pythonModel: the untrained python model (undefined)

        """
        working_dir = self.getWorkingDir()
        if working_dir.startswith("file:/"):
            working_dir = working_dir[5:]
        trainer_path = join(working_dir, "trainer")
        from cntk import combine
        combined_func = combine(value.model, value.loss_function, value.evaluation_function)
        combined_func.save(trainer_path)
        self._set(pythonModelPath=trainer_path)
        self._set(learnerCode=learnerCode)
        return self
