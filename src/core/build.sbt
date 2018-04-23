libraryDependencies ++= Seq(
  // "%%" for scala things, "%" for plain java things
  "com.typesafe" % "config" % "1.3.1",
  "org.apache.logging.log4j" %  "log4j-api"       % "2.8.1" % "provided",
  "org.apache.logging.log4j" %  "log4j-core"      % "2.8.1" % "provided",
  "org.apache.logging.log4j" %% "log4j-api-scala" % "2.8.1" % "provided"
)