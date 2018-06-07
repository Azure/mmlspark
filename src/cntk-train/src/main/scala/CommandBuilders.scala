// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.io.FileNotFoundException
import java.net.URI

import scala.collection.mutable.ListBuffer
import scala.sys.process._
import FileUtilities._
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.SparkSession
import org.slf4j.Logger
import StreamUtilities.using

abstract class CNTKCommandBuilderBase(log: Logger) {
  val command: String
  def arguments(): Seq[String]
  val configs = ListBuffer.empty[BrainScriptConfig]

  var workingDir = new File(".").toURI
  var outputDir: String = ""
  var modelName: String = ""
  var modelOutputDir: String = ""
  var sparkSession: SparkSession = null
  var dataFormat: String = ""

  def setWorkingDir(p: String): this.type = {
    workingDir = new File(p).toURI
    this
  }

  def setModelOutputDir(p: String): this.type = {
    modelOutputDir = p
    this
  }

  def setModelName(p: String): this.type = {
    modelName = p
    this
  }

  def insertBaseConfig(t: String): this.type = {
    configs.insert(0, BrainScriptConfig("baseConfig", Seq(t)))
    this
  }

  def appendOverrideConfig(t: Seq[String]): this.type = {
    configs.append(BrainScriptConfig("overrideConfig", t))
    this
  }

  def setOutputDir(p: String): this.type = {
    outputDir = p
    this
  }

  def setSparkSession(p: SparkSession): this.type = {
    sparkSession = p
    this
  }

  def setDataFormat(d: String): this.type = {
    dataFormat = d
    this
  }

  protected def configToFile(c: BrainScriptConfig): String = {
    val outFolder = new File(workingDir)
    val outFile = new File(s"${outFolder.getAbsolutePath}/${c.name}.cntk")
    if (!outFolder.exists()) {
      log.info(s"Creating directory $outFolder")
      outFolder.mkdirs()
    }
    writeFile(outFile, c.text.mkString("\n"))

    log.info(s"wrote string to ${outFile.toPath}")
    outFile.getAbsolutePath
  }

  def runCommand(): Unit

  protected def printOutput(command: ProcessBuilder): Unit = {
    var result = ProcessUtils.getProcessOutput(log, command)
    log.info(s"Command succeeded with output: $result")
  }
}

class CNTKCommandBuilder(log: Logger, fileBased: Boolean = true) extends CNTKCommandBuilderBase(log) {
  val command = "cntk"
  val arguments = Seq[String]()

   def runCommand(): Unit = {
     val cntkArgs = configs
       .map(c => if (fileBased) s"configFile=${configToFile(c)} " else c.text.mkString(" "))
       .mkString(" ")
     printOutput(command + " " + cntkArgs)
  }
}

trait MPIConfiguration {
  val command = "mpirun"
  // nodename -> workers per node
  def nodeConfig: Map[String, Int]
}

class MPICommandBuilder(log: Logger,
                        gpuMachines: Array[String],
                        hdfsPath: Option[(String, String, String)],
                        fileInputPath: String,
                        username: String) extends CNTKCommandBuilderBase(log) with MPIConfiguration {
  private val defaultNumGPUs = 1

  def nodeConfig: Map[String, Int] = gpuMachines.map(value => {
    val nodeAndGpus = value.split(",")
    (nodeAndGpus(0),
      if (nodeAndGpus.length == 2)
        try {
          nodeAndGpus(1).trim.toInt
        } catch {
          case ex: Exception => defaultNumGPUs
        }
      else defaultNumGPUs)
  }).toMap

  val argName = "-n"
  val arguments = Seq(argName, nodeConfig.head._2.toString)
  val identityKeyException =
    "Please run the passwordless ssh setup:" +
    " mmlspark/tools/hdi/setup-ssh-access.sh\n" +
    "to create a private key.\n" +
    "Identity file not found: "

  protected def nodeConfigToFile(nodeConfig: Map[String, Int]): String = {
    val outFolder = new File(workingDir)
    val outFile = new File(s"${outFolder.getAbsolutePath}/hostfile.txt")
    if (!outFolder.exists()) {
      log.info(s"Creating directory $outFolder")
      outFolder.mkdirs()
    }
    val txt = nodeConfig.map{ case(name, num) => s"$name slots=$num" }.mkString("\n")
    writeFile(outFile, txt)

    log.info(s"wrote node configuration to ${outFile.toPath}")
    outFile.getAbsolutePath
  }

  def runCommand(): Unit = {
    val exportClasspath = "export CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob); "
    val cntkArgs = "cntk " + configs
      .map(c => s"configFile=${configToFile(c)} ")
      .mkString(" ")
    var mpiArgs = " -x CLASSPATH=$CLASSPATH" + s" --hostfile ${nodeConfigToFile(nodeConfig)} "

    val hadoopConf = sparkSession.sparkContext.hadoopConfiguration
    val inputPath = new Path(CNTKLearner.identityLocation)
    // Get the user's home directory
    val userHomePath = System.getProperty("user.home")
    val identityDir = new File(new Path(userHomePath, CNTKLearner.localSSH).toString)
    if (!identityDir.exists()) {
      log.info(s"Creating directory $identityDir")
      identityDir.mkdirs()
    }
    val identity = s"${identityDir.getAbsolutePath}/identity"
    val identityPath = new Path(identity)
    val outputPath = new Path(s"file:///${identityDir.getAbsolutePath}/identity")
    // Copy from wasb to local file
    try {
      using(inputPath.getFileSystem(hadoopConf)) { fs =>
        fs.copyToLocalFile(inputPath, identityPath)
      }.get
    } catch {
      case e: FileNotFoundException =>
        throw new RuntimeException(identityKeyException + e.getMessage, e)
    }

    val primaryNodeName = nodeConfig.head._1
    val gpuUser = s"$username@$primaryNodeName"
    val localDir = workingDir.toString.replaceFirst("file:/+", "/")
    var fileDirStr: String = ""
    val localPrivateKeyPath = s"/home/$username/.ssh/id_rsa"
    val privateKeyPath = s"$gpuUser:$localPrivateKeyPath"

    try {
      val localOutputPath = new Path(localDir, outputDir).toString
      val modelDir = new File(modelOutputDir)
      // Create the directory if it does not exist
      if (!modelDir.exists()) {
        log.info(s"Creating directory $modelDir")
        modelDir.mkdirs()
      }

      // Give less permissive file permissions to the private RSA key
      printOutput(Seq("hdfs", "dfs", "-chmod", "700", outputPath.toString))
      // Copy the private RSA key to the primary GPU machine
      printOutput(
        Seq("scp", "-i", identity, "-o", "StrictHostKeyChecking=no", identity, privateKeyPath))
      // Copy the working directory to the GPU machines
      for ((nodeName,numGPUs) <- nodeConfig) {
        val remoteUser = s"$username@$nodeName"
        printOutput(
          Seq("scp", "-i", identity, "-r", "-o", "StrictHostKeyChecking=no", localDir, s"$remoteUser:$localDir"))
      }

      // Run commands below only if writing input data to HDFS
      if (hdfsPath.isDefined && dataFormat == CNTKLearner.textDataFormat) {
        // Add the mounted directory and all parents if it does not exist so mount will work
        fileDirStr = new URI(hdfsPath.get._2).toString
        if (!fileDirStr.startsWith("/")) {
          fileDirStr = "/" + fileDirStr
        }
        printOutput(Seq("ssh", "-i", identity, gpuUser, "mkdir", "-p", fileDirStr))
        // Create local directory if it does not exist on the driver
        val fileDirFile = new File(fileDirStr)
        if (!fileDirFile.exists()) {
          log.info(s"Creating directory $fileDirFile")
          fileDirFile.mkdirs()
        }
        // Do HDFS merge to local directory
        printOutput(Seq("hdfs", "dfs", "-getmerge", s"${hdfsPath.get._1}/*.txt", fileInputPath))
        // scp the file to the GPU machines
        for ((nodeName,numGPUs) <- nodeConfig) {
          val remoteUser = s"$username@$nodeName"
          printOutput(
            Seq("scp", "-i", identity, "-r", "-o", "StrictHostKeyChecking=no",
              fileInputPath, s"$remoteUser:$fileInputPath"))
        }
      }

      // Validate ssh connection from the primary GPU machine to the rest in the list
      if (nodeConfig.size > 1) {
        for ((nodeName,numGPUs) <- nodeConfig) {
          if (nodeName != primaryNodeName) {
            val remoteUser = s"$username@$nodeName"
            val sshCommand = "''" + s"ssh -o StrictHostKeyChecking=no $remoteUser hostname" + "''"
            printOutput(Seq("ssh", "-i", identity, gpuUser, sshCommand))
          }
        }
      }
      val runMPI = "''" + s"$exportClasspath time $command $mpiArgs $cntkArgs" + "''"
      // Run the mpi command
      printOutput(Seq("ssh", "-i", identity, gpuUser, runMPI))
      // Copy the model back
      val modelOrigin = s"$gpuUser:$modelOutputDir/$modelName"
      printOutput(Seq("scp", "-i", identity, "-o", "StrictHostKeyChecking=no", modelOrigin, modelOutputDir))
    } finally {
      // Cleanup: Remove the private RSA key on the primary GPU machine
      printOutput(Seq("ssh", "-i", identity, gpuUser, "rm", localPrivateKeyPath))
      // Cleanup: Remove each GPU machine's working directory
      for ((nodeName,numGPUs) <- nodeConfig) {
        val remoteUser = s"$username@$nodeName"
        printOutput(Seq("ssh", "-i", identity, remoteUser, "rm", "-r", localDir))
      }
      if (hdfsPath.isDefined) {
        // Cleanup: Remove the HDFS directory
        printOutput(Seq("hdfs", "dfs", "-rm", "-r", s"${hdfsPath.get._1}"))
        // Remove the temporary input data files on the GPU machines (not needed for parquet data)
        if (dataFormat == CNTKLearner.textDataFormat) {
          for ((nodeName,numGPUs) <- nodeConfig) {
            val remoteUser = s"$username@$nodeName"
            printOutput(Seq("ssh", "-i", identity, remoteUser, "rm", "-r", fileDirStr))
          }
        }
      }
    }
  }

}

class PythonMPICommandBuilder(log: Logger,
                              gpuMachines: Array[String],
                              hdfsPath: Option[(String, String, String)],
                              fileInputPath: String,
                              username: String,
                              pythonModelPath: String,
                              learnerCode: String,
                              inputData: InputData,
                              features: String,
                              labels: String) extends CNTKCommandBuilderBase(log) with MPIConfiguration {
  private val defaultNumGPUs = 1

  def nodeConfig: Map[String, Int] = gpuMachines.map(value => {
    val nodeAndGpus = value.split(",")
    (nodeAndGpus(0),
      if (nodeAndGpus.length == 2)
        try {
          nodeAndGpus(1).toInt
        } catch {
          case ex: Exception => defaultNumGPUs
        }
      else defaultNumGPUs)
  }).toMap

  val argName = "-n"
  val arguments = Seq(argName, nodeConfig.head._2.toString)
  val identityKeyException =
    "Please run the passwordless ssh setup: mmlspark/tools/hdi/setup-ssh-keys.sh\n" +
      "  to create a private key.\n" +
      "Identity file not found: "

  def runCommand(): Unit = {
    val mpiArgs = s" --npernode ${nodeConfig.head._2} "

    val hadoopConf = sparkSession.sparkContext.hadoopConfiguration
    val inputPath = new Path("wasb:///MML-GPU/identity")
    // Get the user's home directory
    val userHomePath = System.getProperty("user.home")
    val identityDir = new File(new Path(userHomePath, ".ssh/MML-GPU").toString)
    if (!identityDir.exists()) {
      log.info(s"Creating directory $identityDir")
      identityDir.mkdirs()
    }
    val identity = s"${identityDir.getAbsolutePath}/identity"
    val identityPath = new Path(identity)
    val outputPath = new Path(s"file:///${identityDir.getAbsolutePath}/identity")
    // Copy from wasb to local file
    try {
      using(inputPath.getFileSystem(hadoopConf)) { fs =>
        fs.copyToLocalFile(inputPath, identityPath)
      }.get
    } catch {
      case e: FileNotFoundException =>
        throw new RuntimeException(identityKeyException + e.getMessage, e)
    }

    val localDir = workingDir.toString.replaceFirst("file:/+", "/")
    val modelPath = new Path(localDir, new Path(outputDir, "Models")).toString
    val localOutputPath = new Path(localDir, outputDir).toString

    val pythonTrainer = "distributed_train.py"
    val pathToPythonTrainer = new Path(localDir, pythonTrainer).toString
    val pythonArgs = s" python $pathToPythonTrainer"

    val modelDir = new File(modelPath)
    // Create the directory if it does not exist - note: this also generates the dir for python trainer file
    if (!modelDir.exists()) modelDir.mkdirs()

    // Generate the python file locally
    val featuresShape = inputData.shapes(features)
    val labelsShape = inputData.shapes(labels)
    val (featuresDim, isInputSparse) = (featuresShape.dim, featuresShape.form == CNTKLearner.sparseForm)
    val (labelsDim, isLabelSparse) = (labelsShape.dim, labelsShape.form == CNTKLearner.sparseForm)
    val localModelPath = pythonModelPath.replaceFirst("file:/+", "/")
    val newModelPath = s"$modelPath/ModelOut"
    ModelBuilder.generateModel(localModelPath,
      learnerCode,
      newModelPath,
      featuresDim,
      labelsDim,
      inputData.path,
      isInputSparse,
      isLabelSparse,
      new File(pathToPythonTrainer))

    val nodeName = nodeConfig.head._1
    val gpuUser = s"$username@$nodeName"

    // Give less permissive file permissions to the private RSA key
    printOutput(Seq("hdfs", "dfs", "-chmod", "700", outputPath.toString))
    // Copy the working directory to the GPU machines
    printOutput(
      Seq("scp", "-i", identity, "-r", "-o", "StrictHostKeyChecking=no", localDir, s"$gpuUser:$localDir"))
    // Copy the python model to the GPU machines
    printOutput(
      Seq("scp", "-i", identity, "-r", "-o", "StrictHostKeyChecking=no", localModelPath, s"$gpuUser:$localModelPath"))

    // Run commands below only if writing input data to HDFS
    var fileDirStr: String = ""
    if (hdfsPath.isDefined) {
      // Add the mounted directory and all parents if it does not exist so mount will work
      fileDirStr = new URI(hdfsPath.get._2).toString
      if (!fileDirStr.startsWith("/")) {
        fileDirStr = "/" + fileDirStr
      }
      printOutput(Seq("ssh", "-i", identity, gpuUser, "mkdir", "-p", fileDirStr))
      val mergedInputFile = "merged-input.txt"
      // Create local directory if it does not exist on the driver
      val fileDirFile = new File(fileDirStr)
      if (!fileDirFile.exists()) {
        fileDirFile.mkdirs()
      }
      // Do HDFS merge to local directory
      printOutput(Seq("hdfs", "dfs", "-getmerge", s"${hdfsPath.get._1}/*.txt", fileInputPath))
      // scp the file to the GPU machine
      printOutput(
        Seq("scp", "-i", identity, "-r", "-o", "StrictHostKeyChecking=no",
          fileInputPath, s"$gpuUser:$fileInputPath"))
    }
    // Activate python environment
    val activateCmd = "source \"/home/" + username + "/cntk/activate-cntk\""
    // Run the mpi command - need to activate the environment and run the command in the same script
    val runMPI = "''" + s"$activateCmd; $command $mpiArgs $pythonArgs" + "''"
    printOutput(Seq("ssh", "-i", identity, gpuUser, runMPI))
    // Copy the model back
    val modelOrigin = s"$gpuUser:$modelPath"
    printOutput(Seq("scp", "-i", identity, "-r", "-o", "StrictHostKeyChecking=no", modelOrigin, localOutputPath))
    // Cleanup: Remove the GPU machine's working directory
    printOutput(Seq("ssh", "-i", identity, gpuUser, "rm", "-r", localDir))
    if (hdfsPath.isDefined) {
      // Cleanup: Remove the HDFS directory
      printOutput(Seq("hdfs", "dfs", "-rm", "-r", s"${hdfsPath.get._1}"))
      // Remove the temporary files on the GPU machine
      printOutput(Seq("ssh", "-i", identity, gpuUser, "rm", "-r", fileDirStr))
    }
  }

}
