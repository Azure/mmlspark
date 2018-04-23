// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk.train

import java.io.File
import java.net.URI

import com.microsoft.ml.spark.Readers.implicits._
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.opencv.UnrollImage
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.scalatest.{BeforeAndAfterEach, Suite}

trait TestFileCleanup extends BeforeAndAfterEach {
  this: Suite =>
  var cleanupPath: File
  override def afterEach(): Unit = {
    try super.afterEach() // To be stackable, must call super.afterEach
    finally {
      if (cleanupPath.exists) FileUtils.forceDelete(cleanupPath)
    }
  }
}

class ValidateCntkTrain extends TestBase with TestFileCleanup {

  override var cleanupPath: File = new File(new URI(dir))

  val dummyTrainScript = s"""
command = trainNetwork:testNetwork

precision = "float"; traceLevel = 1 ; deviceId = "auto"

rootDir = ".." ; dataDir = "$$rootDir$$/DataSets/MNIST" ;
outputDir = "./Output" ;

modelPath = "$$outputDir$$/Models/01_OneHidden"

# TRAINING CONFIG
trainNetwork = {
    action = "train"

    BrainScriptNetworkBuilder = {
        labelDim = 1 # number of distinct labels

        # This model returns multiple nodes as a record, which
        # can be accessed using .x syntax.
        model(x) = {
            h1 = DenseLayer {5, activation=ReLU} (x)
            z = LinearLayer {labelDim} (h1)
        }

        # inputs
        features = Input {9}
        labels = Input {labelDim}

        # apply model to features
        out = model (features)

        # loss and error computation
        ce   = CrossEntropyWithSoftmax (labels, out.z)
        errs = ClassificationError (labels, out.z)

        # declare special nodes
        featureNodes    = (features)
        labelNodes      = (labels)
        criterionNodes  = (ce)
        evaluationNodes = (errs)
        outputNodes     = (out.z)
    }

    SGD = {
        epochSize = 60
        minibatchSize = 6
        maxEpochs = 3
        learningRatesPerSample = 0.0001
        momentumAsTimeConstant = 0

        numMBsToShowResult = 500
    }

    reader = {
        readerType = "CNTKTextFormatReader"
        # See ../README.md for details on getting the data (Train-28x28_cntk_text.txt).
        file = "file:///Train-28x28_cntk_text.txt"
        input = {
            features = { dim = 784 ; format = "dense" }
            labels =   { dim = 10  ; format = "dense" }
        }
    }
}

# TEST CONFIG
testNetwork = {
    action = "test"
    minibatchSize = 1024    # reduce this if you run out of memory

    reader = {
        readerType = "CNTKTextFormatReader"
        file = "file:///Test-28x28_cntk_text.txt"
        input = {
            features = { dim = 784 ; format = "dense" }
            labels =   { dim = 10  ; format = "dense" }
        }
    }
}
"""

  val cifarScript = s"""
# ConvNet applied on CIFAR-10 dataset, with no data augmentation.

command = TrainNetwork

precision = "float"; traceLevel = 0 ; deviceId = "auto"

rootDir = "../../.." ; dataDir = "$$rootDir$$/DataSets/CIFAR-10" ;
outputDir = "./Output" ;

TrainNetwork = {
    action = "train"

    BrainScriptNetworkBuilder = {
        imageShape = 32:32:3
        labelDim = 6

        featMean = 128
        featScale = 1/256
        Normalize{m,f} = x => f .* (x - m)

        model = Sequential (
            Normalize {featMean, featScale} :
            ConvolutionalLayer {64, (3:3), pad = true} : ReLU :
            ConvolutionalLayer {64, (3:3), pad = true} : ReLU :
              MaxPoolingLayer {(3:3), stride = (2:2)} :
            ConvolutionalLayer {64, (3:3), pad = true} : ReLU :
            ConvolutionalLayer {64, (3:3), pad = true} : ReLU :
              MaxPoolingLayer {(3:3), stride = (2:2)} :
            DenseLayer {256} : ReLU : Dropout :
            DenseLayer {128} : ReLU : Dropout :
            LinearLayer {labelDim}
        )

        # inputs
        features = Input {imageShape}
        labels   = Input {labelDim}

        # apply model to features
        z = model (features)

        # connect to system
        ce       = CrossEntropyWithSoftmax     (labels, z)
        errs     = ClassificationError         (labels, z)
        top5Errs = ClassificationError         (labels, z, topN=5)  # only used in Eval action

        featureNodes    = (features)
        labelNodes      = (labels)
        criterionNodes  = (ce)
        evaluationNodes = (errs)  # top5Errs only used in Eval
        outputNodes     = (z)
    }

    SGD = {
        epochSize = 0
        minibatchSize = 256

        learningRatesPerSample = 0.0015625*10:0.00046875*10:0.00015625
        momentumAsTimeConstant = 0*20:607.44
        maxEpochs = 30
        L2RegWeight = 0.002
        dropoutRate = 0*5:0.5

        numMBsToShowResult = 100
        parallelTrain = {
            parallelizationMethod = "DataParallelSGD"
            parallelizationStartEpoch = 2  # warm start: don't use 1-bit SGD for first epoch
            distributedMBReading = true
            dataParallelSGD = { gradientBits = 1 }
        }
    }

    reader = {
        readerType = "CNTKTextFormatReader"
        file = "$$DataDir$$/Train_cntk_text.txt"
        randomize = true
        keepDataInMemory = true     # cache all data in memory
        input = {
            features = { dim = 3072 ; format = "dense" }
            labels   = { dim = 6 ;   format = "dense" }
        }
    }
}
"""

  test("Smoke test for training on a classifier") {
    val rawPath = new File(s"${sys.env("DATASETS_HOME")}/Binary/Train", "breast-cancer.train.csv").toString
    val path = normalizePath(rawPath)
    val dataset = session.read
      .option("header", true)
      .option("inferSchema", true)
      .option("nullValue", "?")
      .csv(path)
      .withColumnRenamed("Label", "labels")

    val learner = new CNTKLearner()
      .setBrainScriptText(dummyTrainScript)
      .setParallelTrain(false)
      .setWorkingDirectory(dir)

    val data = dataset.randomSplit(Seq(0.6, 0.4).toArray, 42)
    val trainData = data(0)
    val testData = data(1)

    val model = learner.fit(trainData)
    println(model)
  }

  test("Verify can train on a mix of sparse and dense vectors") {
    val rawPath = new File(s"${sys.env("DATASETS_HOME")}/Binary/Train", "breast-cancer.train.csv").toString
    val path = normalizePath(rawPath)
    val dataset: DataFrame = session.createDataFrame(Seq(
      (0, Vectors.dense(5, 1, 1, 1, 2, 1, 3, 1, 1)),
      (0, Vectors.dense(5, 4, 4, 5, 7, 10, 3, 2, 1)),
      (0, Vectors.dense(3, 1, 1, 1, 2, 2, 3, 1, 1)),
      (0, Vectors.dense(6, 8, 8, 1, 3, 4, 3, 7, 1)),
      (0, Vectors.sparse(9, Array(0, 1, 4, 6, 7, 8), Array(4, 1, 1, 3, 1, 1))),
      (1, Vectors.dense(8, 10, 10, 8, 7, 10, 9, 7, 1)),
      (0, Vectors.sparse(9, Array(1, 2, 3, 4, 5, 6), Array(4, 2, 1, 2, 1, 1))),
      (0, Vectors.dense(1, 1, 1, 1, 1, 1, 3, 1, 1)),
      (0, Vectors.dense(2, 1, 1, 1, 2, 1, 2, 1, 1)),
      (1, Vectors.sparse(9, Array(0, 1, 2, 3, 4, 5, 6, 7, 8), Array(7, 4, 6, 4, 6, 1, 4, 3, 1))),
      (0, Vectors.dense(6, 1, 1, 1, 2, 1, 3, 1, 1)))).toDF("labels", "feats")

    val learner = new CNTKLearner()
      .setBrainScriptText(dummyTrainScript)
      .setParallelTrain(false)
      .setWorkingDirectory(dir)

    val data = dataset.randomSplit(Seq(0.6, 0.4).toArray, 42)
    val trainData = data(0)
    val testData = data(1)

    val model = learner.fit(trainData)
    println(model)
  }

  test("train and eval CIFAR") {
    val trigger = session.sparkContext
    val filesRoot = s"${sys.env("DATASETS_HOME")}/"
    val imagePath = s"$filesRoot/Images/CIFAR"

    val inputCol = "cntk_images"
    val tmpLabel = "labelscol"
    val indexedLabel = "idxlabels"
    val labelCol = "labels"

    val images = session.readImages(imagePath, true)

    // Label annotation: CIFAR is constructed here as
    // 01234-01.png, meaning (len - 5, len - 3) is label
    val pathLen = images.first.getStruct(0).getString(0).length
    val labeledData = images.withColumn(tmpLabel, images("image.path").substr(pathLen - 5, 2).cast("float"))

    // Unroll images into Spark representation
    val unroller = new UnrollImage().setOutputCol(inputCol).setInputCol("image")
    val unrolled = unroller.transform(labeledData).select(inputCol, tmpLabel)

    // Prepare Spark-like DF with known labels
    val ohe = new OneHotEncoder().setInputCol(tmpLabel).setOutputCol(labelCol).setDropLast(false)
    val dataset = ohe.transform(unrolled).select(inputCol, labelCol)
    val learner = new CNTKLearner()
      .setBrainScriptText(cifarScript)
      // Build machine doesn't have GPUs
      .setParallelTrain(false)
      .setWorkingDirectory(dir)

    val model = learner.fit(dataset)
      .setInputCol(inputCol)
      .setOutputCol("out_labels")
      .setOutputNodeIndex(3)

    val result = model.transform(dataset)
    result.take(1)
  }
}
