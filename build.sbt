import java.io.{File, PrintWriter}
import java.net.URL
import org.apache.commons.io.FileUtils
import sbt.ExclusionRule
import sbt.internal.util.ManagedLogger

import scala.xml.{Node => XmlNode, NodeSeq => XmlNodeSeq, _}
import scala.xml.transform.{RewriteRule, RuleTransformer}
import scala.sys.process.Process
import BuildUtils._

val condaEnvName = "mmlspark"
name := "mmlspark"
organization := "com.microsoft.ml.spark"
scalaVersion := "2.12.10"
val sparkVersion = "3.0.1"

//val scalaMajorVersion  = settingKey[String]("scalaMajorVersion")
//scalaMajorVersion  := {scalaVersion.value.split(".".toCharArray).dropRight(0).mkString(".")}
val scalaMajorVersion = 2.12

val excludes = Seq(
  ExclusionRule("org.apache.spark", s"spark-tags_$scalaMajorVersion"),
  ExclusionRule("org.scalatest")
)

libraryDependencies ++= Seq(
  "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.12.3",
  "org.apache.spark" %% "spark-core" % sparkVersion % "compile",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "compile",
  "org.apache.spark" %% "spark-avro" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-tags" % sparkVersion % "test",
  "org.scalatest" %% "scalatest" % "3.0.5" % "test")

libraryDependencies ++= Seq(
  "org.scalactic" %% "scalactic" % "3.0.5",
  "io.spray" %% "spray-json" % "1.3.2",
  "com.microsoft.cntk" % "cntk" % "2.4",
  "org.openpnp" % "opencv" % "3.2.0-1",
  "com.jcraft" % "jsch" % "0.1.54",
  "com.microsoft.cognitiveservices.speech" % "client-sdk" % "1.14.0",
  // TA SDK (latest preview) https://mvnrepository.com/artifact/com.azure/azure-ai-textanalytics/5.1.0-beta.7
  // |
  //
  "com.azure" % "azure-ai-textanalytics" % "5.1.0-beta.7",
  "org.apache.httpcomponents" % "httpclient" % "4.5.6",
  "org.apache.httpcomponents" % "httpmime" % "4.5.6",
  "com.microsoft.ml.lightgbm" % "lightgbmlib" % "3.2.110",
  "com.github.vowpalwabbit" % "vw-jni" % "8.9.1",
  "com.linkedin.isolation-forest" %% "isolation-forest_3.0.0" % "1.0.1",
).map(d => d excludeAll (excludes: _*))

def txt(e: Elem, label: String): String = "\"" + e.child.filter(_.label == label).flatMap(_.text).mkString + "\""

def activateCondaEnv: Seq[String] = {
  if (sys.props("os.name").toLowerCase.contains("windows")) {
    osPrefix ++ Seq("activate", condaEnvName, "&&")
  } else {
    Seq()
    //TODO figure out why this doesent work
    //Seq("/bin/bash", "-l", "-c", "source activate " + condaEnvName, "&&")
  }
}

val omittedDeps = Set(s"spark-core_${scalaMajorVersion}", s"spark-mllib_${scalaMajorVersion}", "org.scala-lang")
// skip dependency elements with a scope
pomPostProcess := { (node: XmlNode) =>
  new RuleTransformer(new RewriteRule {
    override def transform(node: XmlNode): XmlNodeSeq = node match {
      case e: Elem if e.label == "dependency"
        && e.child.exists(child => child.label == "scope") =>
        Comment(
          s""" scoped dependency ${txt(e, "groupId")} % ${txt(e, "artifactId")}
             |% ${txt(e, "version")} % ${txt(e, "scope")} has been omitted """.stripMargin)
      case e: Elem if e.label == "dependency"
        && e.child.exists(child => omittedDeps(child.text)) =>
        Comment(
          s""" excluded dependency ${txt(e, "groupId")} % ${txt(e, "artifactId")}
             |% ${txt(e, "version")} has been omitted """.stripMargin)
      case _ => node
    }
  }).transform(node).head
}

resolvers += "Speech" at "https://mmlspark.blob.core.windows.net/maven/"

val createCondaEnvTask = TaskKey[Unit]("createCondaEnv", "create conda env")
createCondaEnvTask := {
  val s = streams.value
  val hasEnv = Process("conda env list").lineStream.toList
    .map(_.split("\\s+").head).contains(condaEnvName)
  if (!hasEnv) {
    runCmd(Seq("conda", "env", "create", "-f", "environment.yaml"))
  } else {
    println("Found conda env " + condaEnvName)
  }
}

val condaEnvLocation = TaskKey[String]("condaEnvLocation", "get install location of conda env")
condaEnvLocation := {
  val s = streams.value
  createCondaEnvTask.value
  Process("conda env list").lineStream.toList
    .map(_.split("\\s+"))
    .map(l => (l.head, l.reverse.head))
    .filter(p => p._1 == condaEnvName)
    .head._2
}


val cleanCondaEnvTask = TaskKey[Unit]("cleanCondaEnv", "create conda env")
cleanCondaEnvTask := {
  runCmd(Seq("conda", "env", "remove", "--name", condaEnvName, "-y"))
}

val codegenTask = TaskKey[Unit]("codegen", "Generate Code")
codegenTask := {
  (runMain in Test).toTask(" com.microsoft.ml.spark.codegen.CodeGen").value
}

val testgenTask = TaskKey[Unit]("testgen", "Generate Tests")
testgenTask := {
  (runMain in Test).toTask(" com.microsoft.ml.spark.codegen.TestGen").value
}

val genDir = join("target", s"scala-${scalaMajorVersion}", "generated")
val unidocDir = join("target", s"scala-${scalaMajorVersion}", "unidoc")
val pythonSrcDir = join(genDir.toString, "src", "python")
val unifiedDocDir = join(genDir.toString, "doc")
val pythonDocDir = join(unifiedDocDir.toString, "pyspark")
val pythonPackageDir = join(genDir.toString, "package", "python")
val pythonTestDir = join(genDir.toString, "test", "python")
val rSrcDir = join(genDir.toString, "src", "R", "mmlspark")
val rPackageDir = join(genDir.toString, "package", "R")

val pythonizedVersion = settingKey[String]("Pythonized version")
pythonizedVersion := {
  if (version.value.contains("-")) {
    version.value.split("-".head).head + ".dev1"
  } else {
    version.value
  }
}

val rVersion = settingKey[String]("R version")
rVersion := {
  if (version.value.contains("-")) {
    version.value.split("-".head).head
  } else {
    version.value
  }
}

def rCmd(cmd: Seq[String], wd: File, libPath: String): Unit = {
  runCmd(activateCondaEnv ++ cmd, wd, Map("R_LIBS" -> libPath, "R_USER_LIBS" -> libPath))
}

val packageR = TaskKey[Unit]("packageR", "Generate roxygen docs and zip R package")
packageR := {
  createCondaEnvTask.value
  codegenTask.value
  val libPath = join(condaEnvLocation.value, "Lib", "R", "library").toString
  rCmd(Seq("R", "-q", "-e", "roxygen2::roxygenise()"), rSrcDir, libPath)
  rPackageDir.mkdirs()
  zipFolder(rSrcDir, new File(rPackageDir, s"mmlspark-${version.value}.zip"))
}

val testR = TaskKey[Unit]("testR", "Run testthat on R tests")
testR := {
  packageR.value
  publishLocal.value
  val libPath = join(condaEnvLocation.value, "Lib", "R", "library").toString
  rCmd(Seq("R", "CMD", "INSTALL", "--no-multiarch", "--with-keep.source", "mmlspark"), rSrcDir.getParentFile, libPath)
  val testRunner = join("tools", "tests", "run_r_tests.R").getAbsolutePath
  rCmd(Seq("Rscript", testRunner), rSrcDir, libPath)
}

val publishR = TaskKey[Unit]("publishR", "publish R package to blob")
publishR := {
  codegenTask.value
  packageR.value
  val rPackage = rPackageDir.listFiles().head
  singleUploadToBlob(rPackage.toString, rPackage.getName, "rrr")
}

val packagePythonTask = TaskKey[Unit]("packagePython", "Package python sdk")
packagePythonTask := {
  codegenTask.value
  createCondaEnvTask.value
  val destPyDir = join("target", s"scala-${scalaMajorVersion}", "classes", "mmlspark")
  if (destPyDir.exists()) FileUtils.forceDelete(destPyDir)
  FileUtils.copyDirectory(join(pythonSrcDir.getAbsolutePath, "mmlspark"), destPyDir)
  runCmd(
    activateCondaEnv ++
      Seq(s"python", "setup.py", "bdist_wheel", "--universal", "-d", s"${pythonPackageDir.absolutePath}"),
    pythonSrcDir)
}

val installPipPackageTask = TaskKey[Unit]("installPipPackage", "install python sdk")
installPipPackageTask := {
  packagePythonTask.value
  publishLocal.value
  runCmd(
    activateCondaEnv ++ Seq("pip", "install", "-I",
      s"mmlspark-${pythonizedVersion.value}-py2.py3-none-any.whl"),
    pythonPackageDir)
}

val generatePythonDoc = TaskKey[Unit]("generatePythonDoc", "Generate sphinx docs for python")
generatePythonDoc := {
  installPipPackageTask.value
  runCmd(activateCondaEnv ++ Seq("sphinx-apidoc", "-f", "-o", "doc", "."),
    join(pythonSrcDir.toString, "mmlspark"))
  runCmd(activateCondaEnv ++ Seq("sphinx-build", "-b", "html", "doc", "../../../doc/pyspark"),
    join(pythonSrcDir.toString, "mmlspark"))
}

val publishDocs = TaskKey[Unit]("publishDocs", "publish docs for scala and python")
publishDocs := {
  generatePythonDoc.value
  (Compile / unidoc).value
  val html =
    """
      |<html><body><pre style="font-size: 150%;">
      |<a href="pyspark/index.html">pyspark/</u>
      |<a href="scala/index.html">scala/</u>
      |</pre></body></html>
    """.stripMargin
  val scalaDir = join(unifiedDocDir.toString, "scala")
  if (scalaDir.exists()) FileUtils.forceDelete(scalaDir)
  FileUtils.copyDirectory(unidocDir, scalaDir)
  FileUtils.writeStringToFile(join(unifiedDocDir.toString, "index.html"), html, "utf-8")
  uploadToBlob(unifiedDocDir.toString, version.value, "docs")
}

val publishPython = TaskKey[Unit]("publishPython", "publish python wheel")
publishPython := {
  publishLocal.value
  packagePythonTask.value
  singleUploadToBlob(
    join(pythonPackageDir.toString, s"mmlspark-${pythonizedVersion.value}-py2.py3-none-any.whl").toString,
    version.value + s"/mmlspark-${pythonizedVersion.value}-py2.py3-none-any.whl",
    "pip")
}

val testPythonTask = TaskKey[Unit]("testPython", "test python sdk")

testPythonTask := {
  installPipPackageTask.value
  testgenTask.value
  runCmd(
    activateCondaEnv ++ Seq("python",
      "-m",
      "pytest",
      "--cov=mmlspark",
      "--junitxml=../../../../python-test-results.xml",
      "--cov-report=xml",
      "mmlsparktest"
    ),
    new File(s"target/scala-${scalaMajorVersion}/generated/test/python/")
  )
}

val getDatasetsTask = TaskKey[Unit]("getDatasets", "download datasets used for testing")
val datasetName = "datasets-2020-08-27.tgz"
val datasetUrl = new URL(s"https://mmlspark.blob.core.windows.net/installers/$datasetName")
val datasetDir = settingKey[File]("The directory that holds the dataset")
datasetDir := {
  join(target.value.toString, s"scala-${scalaMajorVersion}", "datasets", datasetName.split(".".toCharArray.head).head)
}

getDatasetsTask := {
  val d = datasetDir.value.getParentFile
  val f = new File(d, datasetName)
  if (!d.exists()) d.mkdirs()
  if (!f.exists()) {
    FileUtils.copyURLToFile(datasetUrl, f)
    UnzipUtils.unzip(f, d)
  }
}

val genBuildInfo = TaskKey[Unit]("genBuildInfo", "generate a build info file")
genBuildInfo := {
  val buildInfo =
    s"""
       |MMLSpark Build and Release Information
       |---------------
       |
       |### Maven Coordinates
       | `${organization.value}:${name.value}_${scalaMajorVersion}:${version.value}`
       |
       |### Maven Resolver
       | `https://mmlspark.azureedge.net/maven`
       |
       |### Documentation Pages:
       |[Scala Documentation](https://mmlspark.blob.core.windows.net/docs/${version.value}/scala/index.html)
       |[Python Documentation](https://mmlspark.blob.core.windows.net/docs/${version.value}/pyspark/index.html)
       |
    """.stripMargin
  val infoFile = join("target", "Build.md")
  if (infoFile.exists()) FileUtils.forceDelete(infoFile)
  FileUtils.writeStringToFile(infoFile, buildInfo, "utf-8")
}

val setupTask = TaskKey[Unit]("setup", "set up library for intellij")
setupTask := {
  (Compile / compile).toTask.value
  (Test / compile).toTask.value
  getDatasetsTask.value
}

val publishBlob = TaskKey[Unit]("publishBlob", "publish the library to mmlspark blob")
publishBlob := {
  publishM2.value
  val scalaVersionSuffix = scalaVersion.value.split(".".toCharArray.head).dropRight(1).mkString(".")
  val nameAndScalaVersion = s"${name.value}_$scalaVersionSuffix"

  val localPackageFolder = join(
    Seq(new File(new URI(Resolver.mavenLocal.root)).getAbsolutePath)
      ++ organization.value.split(".".toCharArray.head)
      ++ Seq(nameAndScalaVersion, version.value): _*).toString

  val blobMavenFolder = organization.value.replace(".", "/") +
    s"/$nameAndScalaVersion/${version.value}"
  uploadToBlob(localPackageFolder, blobMavenFolder, "maven")
}

val release = TaskKey[Unit]("release", "publish the library to mmlspark blob")
release := Def.taskDyn {
  val v = isSnapshot.value
  if (!v) {
    Def.task {
      sonatypeBundleRelease.value
    }
  } else {
    Def.task {
      "Not a release"
    }
  }
}

val publishBadges = TaskKey[Unit]("publishBadges", "publish badges to mmlspark blob")
publishBadges := {
  def enc(s: String): String = {
    s.replaceAllLiterally("_", "__").replaceAllLiterally(" ", "_").replaceAllLiterally("-", "--")
  }

  def uploadBadge(left: String, right: String, color: String, filename: String): Unit = {
    val badgeDir = join(baseDirectory.value.toString, "target", "badges")
    if (!badgeDir.exists()) badgeDir.mkdirs()
    runCmd(Seq("curl",
      "-o", join(badgeDir.toString, filename).toString,
      s"https://img.shields.io/badge/${enc(left)}-${enc(right)}-${enc(color)}"))
    singleUploadToBlob(
      join(badgeDir.toString, filename).toString,
      s"badges/$filename", "icons", extraArgs = Seq("--content-cache-control", "no-cache"))
  }

  uploadBadge("master version", version.value, "blue", "master_version3.svg")
}

val settings = Seq(
  (scalastyleConfig in Test) := baseDirectory.value / "scalastyle-test-config.xml",
  logBuffered in Test := false,
  buildInfoKeys := Seq[BuildInfoKey](
    name, version, scalaVersion, sbtVersion,
    baseDirectory, datasetDir, pythonizedVersion, rVersion),
  parallelExecution in Test := false,
  test in assembly := {},
  assemblyMergeStrategy in assembly := {
    case PathList("META-INF", xs@_*) => MergeStrategy.discard
    case x => MergeStrategy.first
  },
  assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false),
  buildInfoPackage := "com.microsoft.ml.spark.build")

lazy val mmlspark = (project in file("."))
  .enablePlugins(BuildInfoPlugin)
  .enablePlugins(ScalaUnidocPlugin)
  .settings(settings: _*)

import xerial.sbt.Sonatype._

sonatypeProjectHosting := Some(
  GitHubHosting("Azure", "MMLSpark", "mmlspark-support@microsot.com"))
homepage := Some(url("https://github.com/Azure/mmlspark"))
developers := List(
  Developer("mhamilton723", "Mark Hamilton",
    "mmlspark-support@microsoft.com", url("https://github.com/mhamilton723")),
  Developer("imatiach-msft", "Ilya Matiach",
    "mmlspark-support@microsoft.com", url("https://github.com/imatiach-msft")),
  Developer("drdarshan", "Sudarshan Raghunathan",
    "mmlspark-support@microsoft.com", url("https://github.com/drdarshan"))
)

licenses += ("MIT", url("https://github.com/Azure/mmlspark/blob/master/LICENSE"))
publishMavenStyle := true

credentials += Credentials("Sonatype Nexus Repository Manager",
  "oss.sonatype.org",
  Secrets.nexusUsername,
  Secrets.nexusPassword)

pgpPassphrase := Some(Secrets.pgpPassword.toCharArray)
pgpSecretRing := {
  val temp = File.createTempFile("secret", ".asc")
  new PrintWriter(temp) {
    write(Secrets.pgpPrivate);
    close()
  }
  temp
}
pgpPublicRing := {
  val temp = File.createTempFile("public", ".asc")
  new PrintWriter(temp) {
    write(Secrets.pgpPublic);
    close()
  }
  temp
}

dynverSonatypeSnapshots in ThisBuild := true
dynverSeparator in ThisBuild := "-"
publishTo := sonatypePublishToBundle.value

// Break Cache - 1
