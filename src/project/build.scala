// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

import sbt._
import Keys._

import sys.process.Process

import sbtassembly.AssemblyKeys._
import sbtassembly.AssemblyPlugin.autoImport.assembly
import sbtunidoc.ScalaUnidocPlugin.autoImport.ScalaUnidoc
import sbtunidoc.BaseUnidocPlugin.autoImport.unidoc

object Extras {

  private def env(varName: String, default: String = "") =
    sys.env.getOrElse(varName,
                      if (default != null) default
                      else sys.error(s"Missing $$$varName environment variable"))

  // get the version from $MML_VERSION, or from `show-version`
  def mmlVer = sys.env.getOrElse("MML_VERSION",
                                 Process("../tools/runme/show-version").!!.trim)

  def defaultOrg = "com.microsoft.ml.spark"
  def scalaVer = env("SCALA_FULL_VERSION", null)
  def sparkVer = env("SPARK_VERSION", null)
  def cntkVer  = if (env("CNTK_VERSION", null) == "2.2") "2.2.0.1"
                 else sys.error("Time to remove the CNTK 2.2.0.1 hack!")

  def commonLibs = Seq(
    "org.apache.spark"   %% "spark-core"   % sparkVer % "provided",
    "org.apache.spark"   %% "spark-mllib"  % sparkVer % "provided",
    "org.scalatest"      %% "scalatest"    % "3.0.0"  % "provided",
    // should include these things in the distributed jar
    "io.spray"           %% "spray-json"   % "1.3.2",
    "com.microsoft.cntk"  % "cntk"         % cntkVer,
    "org.openpnp"         % "opencv"       % "3.2.0-1",
    "org.tensorflow"      % "tensorflow"   % "1.5.0-rc0"
    // needed for wasb access, but it collides with the version that comes with Spark,
    // so it gets installed manually for now (see "tools/config.sh")

    // "org.apache.hadoop"   % "hadoop-azure" % "2.7.3"
    )
  def overrideLibs = Seq(
    // spark wants 2.2.6, but we don't use its tests anyway
    "org.scalatest" %% "scalatest" % "3.0.0" % "provided"
    )

  def artifactsDir = file(env("BUILD_ARTIFACTS", "../BuildArtifacts"))
  def testsDir     = file(env("TEST_RESULTS", "../TestResults"))
  def mavenDir     = artifactsDir / "packages" / "m2"
  def docsDir      = artifactsDir / "docs" / "scala"
  val topDocHtml   = file(".") / "project" / "top-doc.html"

  def scalacOpts = Seq(
    "-encoding", "UTF-8",
    // Explain warnings, optimize
    "-deprecation", "-unchecked", "-feature", "-optimise",
    "-Xfatal-warnings", "-Xlint", // all warnings
    // -Y* are Scala options
    "-Yno-adapted-args", // "-Ywarn-adapted-args",
    "-Ywarn-dead-code",
    "-Ywarn-numeric-widen",
    "-Ywarn-value-discard"
    // this leads to problems sometimes: "-Yinline-warnings"
  )

  // Some convenience commands
  val sectionPrefix =
    if (env("BUILDMODE") == "server") "##[section]SBT: " else "===>>> SBT: "
  def addCommands(st: State, cmds: String*): State =
    st.copy(remainingCommands =
              cmds.map(Exec(_, None, None)).toList ++ st.remainingCommands)
  def newCommands = Seq(
    Command.single("cd") { (st, arg) =>
      addCommands(st, s"project $arg") },
    Command.args("echo", "<str>") { (st, args) =>
      println(args.map(s => if (s == "<br>") "\n" else s).mkString(" ")); st },
    Command.single("show-section") { (st, arg) =>
      println("\n" + sectionPrefix + arg); st },
    Command.single("noisy-command") { (st, cmd) =>
      addCommands(st, s"show-section $cmd", cmd) },
    Command.single("on-all-subs") { (st, cmd) =>
      addCommands(st, SubProjects.all.map("noisy-command " + _ + "/" + cmd): _*) },
    Command.command("full-build") { st =>
      val steps = Seq(if (env("PUBLISH") == "all") "update" else null,
                      "run-scalastyle",
                      "compile",
                      if (testSpec == "none") null else "test:compile",
                      "package",
                      if (testSpec == "none") null else "on-all-subs test",
                      "codegen/run",
                      "publish",
                      "unidoc")
      addCommands(st, steps.filter(_ != null).map("noisy-command " + _): _*) }
  ) ++ ScalaStyleExtras.commands

  // Utilities for sub-project sbt files
  def noJar = Seq(Keys.`package` := file(""))

  // Translate $TESTS to command-line arguments
  val testSpec = env("TESTS", "-extended")
  def testOpts =
    // Generate JUnit-style test result files
    Seq(testOptions in (ThisBuild, Test) +=
          Tests.Argument("-u", testsDir.toString())) ++
    (if (testSpec == "all" || testSpec == "none") Seq()
     else testSpec.split(",").map { spec =>
       testOptions in (ThisBuild, Test) +=
         Tests.Argument(if (spec.substring(0,1) == "+") "-n" else "-l",
                        "com.microsoft.ml.spark.test.tags." +
                          spec.substring(1)) })

  def defaultSettings = Seq(
    // Common stuff: defaults for all subprojects
    scalaVersion in ThisBuild := scalaVer,
    organization in ThisBuild := defaultOrg,
    libraryDependencies in ThisBuild ++= commonLibs,
    dependencyOverrides in ThisBuild ++= overrideLibs,
    scalacOptions in ThisBuild ++= scalacOpts,
    scalacOptions in (Compile, doc) += "-groups",
    scalacOptions in (Compile, doc) ++= Seq("-doc-root-content", topDocHtml.getPath()),
    // Don't run tests in parallel, and fork subprocesses for them
    parallelExecution in (ThisBuild, Test) := false,
    fork in (ThisBuild, Test) := true,
    // Assembly options
    aggregate in assembly := false,
    aggregate in publish  := false,
    test in assembly := {},
    // Documentation settings
    autoAPIMappings in ThisBuild := true,
    target in (ScalaUnidoc, unidoc) := docsDir,
    // Ctrl+C kills a running job, not sbt
    cancelable in ThisBuild := true,
    // No verbose logs during update
    logLevel in (ThisBuild, update) := Level.Warn,
    // Fewer errors to display (the default is 100)
    maxErrors in ThisBuild := 20,
    // Show stack traces up to the first SBT stack frame
    traceLevel in ThisBuild := 0,
    // Stamp the jar manifests with the build info
    packageOptions in (Compile, packageBin) +=
      Package.ManifestAttributes(
        "MMLBuildInfo" -> env("MML_BUILD_INFO", "(direct sbt build, no info collected)")),
    // For convenience, import the main package in a scala console
    initialCommands in (ThisBuild, console) := "import com.microsoft.ml.spark._",
    // Use the above commands
    commands in ThisBuild ++= newCommands
    ) ++ testOpts

  def rootSettings =
    defaultSettings ++
    noJar ++ // no toplevel jar
    // With this we get:
    //   mmlspark_2.11-$ver-assembly.jar{,.md5,.sha1}
    //   mmlspark_2.11-$ver.pom{,.md5,.sha1}
    //   mmlspark_2.11-$ver{,-javadoc,-sources}.jar{,.md5,.sha1}
    // the first are the combined jar, and the second are the needed pom files.
    // The third all look empty and discardable.  Without this, we get the same
    // structure, except it seems that it tries to write both the empty jar and
    // the combined jar onto the same mmlspark_2.11-$ver.jar{,.md5,.sha1} files,
    // spitting up a warning, and sometimes the result is the combined jar and
    // sometimes it's the empty (probably the above empty jar with the same no
    // "-assembly" name).  Later in the build we discard the junk files, and
    // leave only the combined one.
    Seq(artifact in (Compile, assembly) :=
          (artifact in (Compile, assembly)).value.withClassifier(Some("assembly"))) ++
    addArtifact(artifact in (Compile, assembly), assembly) ++
    Seq(
      // This creates a maven structure, which we upload to azure storage later
      publishTo := Some(Resolver.file("file", mavenDir)),
      // In case we need to add more stuff to the uber-jar, use this:
      // unmanagedResourceDirectories in Compile += artifactsDir / "more",
      publishArtifact in Test := false,
      publishMavenStyle := true,
      // Remove the "scala-library" dependency
      autoScalaLibrary := false,
      // Don't include things we depend on (we leave the dependency in the POM)
      assemblyOption in assembly :=
        (assemblyOption in assembly).value.copy(
          includeScala = false, includeDependency = false),
      pomPostProcess := { n: scala.xml.Node =>
        import scala.xml._, scala.xml.transform._
        new RuleTransformer(new RewriteRule {
          override def transform(n: Node) =
            // Filter out things that shouldn't be a dependency: things that
            // have "<scope>provided</scope>", or "<optional>true</optional>".
            // The latter is generated in meta.sbt for toplevel dependencies.
            if (n.label == "dependency" &&
                  (n.child.contains(<optional>true</optional>) ||
                     n.child.contains(<scope>provided</scope>)))
              Seq.empty
            else if (n.label == "repositories")
              // Deduplicate repo entries, since we get one for each subproject
              <repositories>{ n.child.distinct }</repositories>
            else
              Seq(n)
        }).transform(Seq(pomPostProcess.value.apply(n))).head
      },
      // Show the current project in the prompt
      shellPrompt in ThisBuild := (st => {
        val ex = Project.extract(st)
        val proj = ex.currentRef.project
        val root = ex.rootProject(ex.currentRef.build)
        s"${if (proj == root) "" else root+"/"}${proj}> "
      }),
      // Use the same history path for everything instead of per project files
      historyPath in ThisBuild := Some((target in LocalRootProject).value / ".history")
    )

  LibraryCheck() // invoke the library checker

}
