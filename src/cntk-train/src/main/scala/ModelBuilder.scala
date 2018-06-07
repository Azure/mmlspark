// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.io.File

object ModelBuilder {

  private def toPyString(bool: Boolean): String = {
    if (bool) "True" else "False"
  }

  def generateModel(inputTrainerPath: String,
                    learnerCode: String,
                    outputModelPath: String,
                    inputDim: Int,
                    labelDim: Int,
                    trainingData: String,
                    isInputSparse: Boolean,
                    isLabelSparse: Boolean,
                    pythonFilePath: File): Unit = {
    val model =
      s"""
         |import cntk
         |from cntk import Trainer, cntk_py, load_model
         |from cntk import data_parallel_distributed_learner, block_momentum_distributed_learner, Communicator
         |from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs
         |from cntk.train.training_session import *
         |from cntk.logging import *
         |from cntk.debugging import *
         |from cntk.cntk_py import Function
         |
         |# autogenerate these
         |trainer_input_path = \"$inputTrainerPath\"
         |model_output_path = \"$outputModelPath\"
         |train_data = \"$trainingData\"
         |input_dim = $inputDim
         |label_dim = $labelDim
         |is_input_sparse = ${toPyString(isInputSparse)}
         |is_label_sparse = ${toPyString(isLabelSparse)}
         |features_name = 'features'
         |labels_name = 'labels'
         |
         |combined_trainer = load_model(trainer_input_path)
         |model = combined_trainer.outputs[0].owner
         |loss_function = combined_trainer.outputs[1].owner
         |evaluation_function = combined_trainer.outputs[2].owner
         |
         |epoch_size = 50000
         |max_epochs = 5
         |minibatch_size = 64
         |num_quantization_bits = 32
         |block_size = None
         |warm_up = False
         |
         |# learner code can override the epoch_size, max_epochs and minibatch size as well as other settings above
         |$learnerCode
         |
         |max_samples = epoch_size * max_epochs
         |
         |learner = None
         |if block_size != None:
         |    learner = block_momentum_distributed_learner(local_learner, block_size=block_size)
         |else:
         |    learner = data_parallel_distributed_learner(local_learner,
         |                                              num_quantization_bits=num_quantization_bits,
         |                                              distributed_after=warm_up)
         |num_mbs_per_log = None
         |log_to_file = None
         |gen_heartbeat = False
         |
         |progress_printer = ProgressPrinter(
         |    freq=num_mbs_per_log,
         |    tag='Training',
         |    log_to_file=log_to_file,
         |    gen_heartbeat=gen_heartbeat,
         |    num_epochs=max_epochs)
         |
         |trainer = Trainer(model, (loss_function, evaluation_function), learner, progress_printer)
         |
         |mbs = MinibatchSource(CTFDeserializer(train_data, StreamDefs(
         |            features  = StreamDef(field=features_name, shape=input_dim, is_sparse=is_input_sparse),
         |            labels = StreamDef(field=labels_name, shape=label_dim, is_sparse=is_label_sparse)
         |        )), randomize=False, max_samples = max_samples)
         |
         |features = loss_function.find_by_name(features_name)
         |labels = loss_function.find_by_name(labels_name)
         |
         |input_map = {
         |    features: mbs.streams.features,
         |    labels: mbs.streams.labels
         |}
         |
         |session = cntk.training_session(
         |    trainer = trainer,
         |    mb_source = mbs,
         |    model_inputs_to_streams = input_map,
         |    mb_size = minibatch_size,
         |    progress_frequency = epoch_size,
         |    checkpoint_config = CheckpointConfig(filename=model_output_path, restore=False)
         |)
         |session.train()
         |
         |#finalize the distributed learning
         |cntk.distributed.Communicator.finalize()
         """.stripMargin
    FileUtilities.writeFile(pythonFilePath, model)
  }
}
