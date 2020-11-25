![MMLSpark](https://mmlspark.azureedge.net/icons/mmlspark.svg)

# Microsoft Machine Learning for Apache Spark

[![Build Status](https://msazure.visualstudio.com/Cognitive%20Services/_apis/build/status/Azure.mmlspark?branchName=master)](https://msazure.visualstudio.com/Cognitive%20Services/_build/latest?definitionId=83120&branchName=master) [![codecov](https://codecov.io/gh/Azure/mmlspark/branch/master/graph/badge.svg)](https://codecov.io/gh/Azure/mmlspark) [![Gitter](https://badges.gitter.im/Microsoft/MMLSpark.svg)](https://gitter.im/Microsoft/MMLSpark?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) 

[![Release Notes](https://img.shields.io/badge/release-notes-blue)](https://github.com/Azure/mmlspark/releases) [![Scala Docs](https://img.shields.io/static/v1?label=api%20docs&message=scala&color=blue&logo=scala)](https://mmlspark.blob.core.windows.net/docs/1.0.0-rc3/scala/index.html#package) [![PySpark Docs](https://img.shields.io/static/v1?label=api%20docs&message=python&color=blue&logo=python)](https://mmlspark.blob.core.windows.net/docs/1.0.0-rc3/pyspark/index.html) [![Academic Paper](https://img.shields.io/badge/academic-paper-7fdcf7)](https://arxiv.org/abs/1810.08744)

[![Version](https://img.shields.io/badge/version-1.0.0--rc3-blue)](https://github.com/Azure/mmlspark/releases) [![Snapshot Version](https://mmlspark.blob.core.windows.net/icons/badges/master_version3.svg)](#sbt) 


MMLSpark is an ecosystem of tools aimed towards expanding the distributed computing framework
[Apache Spark](https://github.com/apache/spark) in several new directions. 
MMLSpark adds many deep learning and data science tools to the Spark ecosystem,
including seamless integration of Spark Machine Learning pipelines with [Microsoft Cognitive Toolkit
(CNTK)](https://github.com/Microsoft/CNTK), [LightGBM](https://github.com/Microsoft/LightGBM) and
[OpenCV](http://www.opencv.org/). These tools enable powerful and highly-scalable predictive and analytical models
for a variety of datasources.

MMLSpark also brings new networking capabilities to the Spark Ecosystem. With the HTTP on Spark project, users 
can embed **any** web service into their SparkML models. In this vein, MMLSpark provides easy to use 
SparkML transformers for a wide variety of [Microsoft Cognitive Services](https://azure.microsoft.com/en-us/services/cognitive-services/). For production grade deployment, the Spark Serving project enables high throughput,
sub-millisecond latency web services, backed by your Spark cluster.

MMLSpark requires Scala 2.11, Spark 2.4+, and Python 3.5+.
See the API documentation [for
Scala](https://mmlspark.blob.core.windows.net/docs/1.0.0-rc3/scala/index.html#package) and [for
PySpark](https://mmlspark.blob.core.windows.net/docs/1.0.0-rc3/pyspark/index.html).

<details>
<summary><strong><em>Table of Contents</em></strong></summary>

-   [Notable features](#notable-features)
-   [A short example](#a-short-example)
-   [Setup and installation](#setup-and-installation)
    -   [Docker](#docker)
    -   [GPU VM Setup](#gpu-vm-setup)
    -   [Spark package](#spark-package)
    -   [Python](#python)
    -   [Databricks cloud](#databricks-cloud)
    -   [SBT](#sbt)
    -   [Building from source](#building-from-source)
-   [Blogs and Publications](#blogs-and-publications)
-   [Contributing & feedback](#contributing--feedback)
-   [Other relevant projects](#other-relevant-projects)

</details>

## Projects

|<img width="800" src="https://mmlspark.blob.core.windows.net/graphics/Readme/vw-blue-dark-orange.svg"> |  <img width="800"  src="https://mmlspark.blob.core.windows.net/graphics/Readme/cog_services_on_spark_2.svg"> | <img width="800"  src="https://mmlspark.blob.core.windows.net/graphics/Readme/decision_tree_recolor.png"> | <img width="800" src="https://mmlspark.blob.core.windows.net/graphics/Readme/mmlspark_serving_recolor.svg"> |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| **Vowpal Wabbit on Spark**  | **The Cognitive Services on Spark**| **LightGBM on Spark** |  **Spark Serving** |
| Fast, Sparse, and Effective Text Analytics | Leverage the Microsoft Cognitive Services at Unprecedented Scales in your existing SparkML pipelines | Train Gradient Boosted Machines with LightGBM   | Serve any Spark Computation as a Web Service with Sub-Millisecond Latency |

|<img width="800" src="https://mmlspark.blob.core.windows.net/graphics/Readme/microservice_recolor.png"> |<img width="800" src="https://mmlspark.blob.core.windows.net/graphics/Readme/distributed_deep_recolor.png"> | <img width="800"  src="https://mmlspark.blob.core.windows.net/graphics/Readme/LIME.svg">|  <img width="800"  src="https://mmlspark.blob.core.windows.net/graphics/Readme/bindings.png"> |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|  **HTTP on Spark** | **CNTK on Spark** |  **Lime on Spark**| **Spark Binding Autogeneration** |
| An Integration Between Spark and the HTTP Protocol, enabling Distributed Microservice Orchestration|Distributed Deep Learning with the Microsoft Cognitive Toolkit | Distributed, Model Agnostic, Interpretations for Classifiers | Automatically Generate Spark bindings for PySpark and SparklyR|

| <img width="150" src="https://mmlspark.blob.core.windows.net/graphics/emails/isolation forest 3.svg"> |<img width="150" src="https://mmlspark.blob.core.windows.net/graphics/emails/cyberml.svg">   | <img width="150" src="https://mmlspark.blob.core.windows.net/graphics/emails/conditional_knn.svg">  |
|:--:|:--:|:--:|
|  **Isolation Forest on Spark**  | **CyberML** | **Conditional KNN**  |
|  Distributed Nonlinear Outlier Detection | Machine Learning Tools for Cyber Security | Scalable KNN Models with Conditional Queries | 



## Examples

-   Create a deep image classifier with transfer learning ([example 9])
-   Fit a LightGBM classification or regression model on a biochemical dataset
    ([example 3]), to learn more check out the [LightGBM documentation
    page](docs/lightgbm.md).
-   Deploy a deep network as a distributed web service with [MMLSpark
    Serving](docs/mmlspark-serving.md)
-   Use web services in Spark with [HTTP on Apache Spark](docs/http.md)
-   Use Bi-directional LSTMs from Keras for medical entity extraction
    ([example 8])
-   Create a text analytics system on Amazon book reviews ([example 4])
-   Perform distributed hyperparameter tuning to identify Breast Cancer
    ([example 5])
-   Easily ingest images from HDFS into Spark `DataFrame` ([example 6])
-   Use OpenCV on Spark to manipulate images ([example 7])
-   Train classification and regression models easily via implicit featurization
    of data ([example 1])
-   Train and evaluate a flight delay prediction system ([example 2])

See our [notebooks](notebooks/samples/) for all examples.

[example 1]: notebooks/samples/Classification%20-%20Adult%20Census.ipynb "Adult Census Income Training"

[example 2]: notebooks/samples/Regression%20-%20Flight%20Delays.ipynb "Regression Example with Flight Delay Dataset"

[example 3]: notebooks/samples/LightGBM%20-%20Quantile%20Regression%20for%20Drug%20Discovery.ipynb "Quantile Regression with LightGBM"

[example 4]: notebooks/samples/TextAnalytics%20-%20Amazon%20Book%20Reviews.ipynb "Amazon Book Reviews - TextFeaturizer"

[example 5]: notebooks/samples/HyperParameterTuning%20-%20Fighting%20Breast%20Cancer.ipynb "Hyperparameter Tuning with MMLSpark"

[example 6]: notebooks/samples/DeepLearning%20-%20CIFAR10%20Convolutional%20Network.ipynb "CIFAR10 CNTK CNN Evaluation"

[example 7]: notebooks/samples/OpenCV%20-%20Pipeline%20Image%20Transformations.ipynb "Pipeline Image Transformations"

[example 8]: notebooks/samples/DeepLearning%20-%20BiLSTM%20Medical%20Entity%20Extraction.ipynb "Medical Entity Extraction"

[example 9]: notebooks/samples/DeepLearning%20-%20Flower%20Image%20Classification.ipynb "Deep Flower Classification"

[example 10]: notebooks/gpu/DeepLearning%20-%20Distributed%20CNTK%20training.ipynb "CIFAR10 CNTK CNN Training"

## A short example

Below is an excerpt from a simple example of using a pre-trained CNN to
classify images in the CIFAR-10 dataset.  View the whole source code in notebook [example 9].

```python
...
import mmlspark
# Initialize CNTKModel and define input and output columns
cntkModel = mmlspark.cntk.CNTKModel() \
  .setInputCol("images").setOutputCol("output") \
  .setModelLocation(modelFile)
# Train on dataset with internal spark pipeline
scoredImages = cntkModel.transform(imagesWithLabels)
...
```

See [other sample notebooks](notebooks/samples/) as well as the MMLSpark
documentation for [Scala](http://mmlspark.azureedge.net/docs/scala/) and
[PySpark](http://mmlspark.azureedge.net/docs/pyspark/).

## Setup and installation

### Python

To try out MMLSpark on a Python (or Conda) installation you can get Spark
installed via pip with `pip install pyspark`.  You can then use `pyspark` as in
the above example, or from python:

```python
import pyspark
spark = pyspark.sql.SparkSession.builder.appName("MyApp") \
            .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc3") \
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
            .getOrCreate()
import mmlspark
```

### SBT

If you are building a Spark application in Scala, add the following lines to
your `build.sbt`:

```scala
resolvers += "MMLSpark" at "https://mmlspark.azureedge.net/maven"
libraryDependencies += "com.microsoft.ml.spark" %% "mmlspark" % "1.0.0-rc3"

```

### Spark package

MMLSpark can be conveniently installed on existing Spark clusters via the
`--packages` option, examples:

```bash
spark-shell --packages com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc3
pyspark --packages com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc3
spark-submit --packages com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc3 MyApp.jar
```

This can be used in other Spark contexts too. For example, you can use MMLSpark
in [AZTK](https://github.com/Azure/aztk/) by [adding it to the
`.aztk/spark-defaults.conf`
file](https://github.com/Azure/aztk/wiki/PySpark-on-Azure-with-AZTK#optional-set-up-mmlspark).

### Databricks

To install MMLSpark on the [Databricks
cloud](http://community.cloud.databricks.com), create a new [library from Maven
coordinates](https://docs.databricks.com/user-guide/libraries.html#libraries-from-maven-pypi-or-spark-packages)
in your workspace.

For the coordinates use: `com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc3` 
with the resolver: `https://mmlspark.azureedge.net/maven`. Ensure this library is
attached to your target cluster(s).

Finally, ensure that your Spark cluster has at least Spark 2.4 and Scala 2.11.

You can use MMLSpark in both your Scala and PySpark notebooks. To get started with our example notebooks import the following databricks archive:

`https://mmlspark.blob.core.windows.net/dbcs/MMLSparkExamplesv1.0.0-rc3.dbc`

### Apache Livy

To install MMLSpark from within a Jupyter notebook served by Apache Livy the following configure magic can be used. You will need to start a new session after this configure cell is executed.

Excluding certain packages from the library may be necessary due to current issues with Livy 0.5

```javascript
%%configure -f
{
    "name": "mmlspark",
    "conf": {
        "spark.jars.packages": "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc3",
        "spark.jars.repositories": "https://mmlspark.azureedge.net/maven",
        "spark.jars.excludes": "org.scala-lang:scala-reflect,org.apache.spark:spark-tags_2.11,org.scalactic:scalactic_2.11,org.scalatest:scalatest_2.11"
    }
}
```

### Docker

The easiest way to evaluate MMLSpark is via our pre-built Docker container.  To
do so, run the following command:

```bash
docker run -it -p 8888:8888 -e ACCEPT_EULA=yes mcr.microsoft.com/mmlspark/release
```

Navigate to <http://localhost:8888/> in your web browser to run the sample
notebooks.  See the [documentation](docs/docker.md) for more on Docker use.

> To read the EULA for using the docker image, run \\
> `docker run -it -p 8888:8888 mcr.microsoft.com/mmlspark/release eula`

### GPU VM Setup

MMLSpark can be used to train deep learning models on GPU nodes from a Spark
application.  See the instructions for [setting up an Azure GPU
VM](docs/gpu-setup.md).



### Building from source

MMLSpark has recently transitioned to a new build infrastructure. 
For detailed developer docs please see the [Developer Readme](docs/developer-readme.md)

If you are an existing mmlspark developer, you will need to reconfigure your 
development setup. We now support platform independent development and 
better integrate with intellij and SBT.
 If you encounter issues please reach out to our support email!

### R (Beta)

To try out MMLSpark using the R autogenerated wrappers [see our
instructions](docs/R-setup.md).  Note: This feature is still under development
and some necessary custom wrappers may be missing.

## Papers

- [Large Scale Intelligent Microservices](https://arxiv.org/abs/2009.08044)

- [Conditional Image Retrieval](https://arxiv.org/abs/2007.07177)

- [MMLSpark: Unifying Machine Learning Ecosystems at Massive Scales](https://arxiv.org/abs/1810.08744)

- [Flexible and Scalable Deep Learning with MMLSpark](https://arxiv.org/abs/1804.04031)

## Learn More

-   Visit our [website].

-   Watch our keynote demos at [the Spark+AI Summit 2019], [the Spark+AI European Summit 2018], and [the Spark+AI Summit 2018].

-   See how MMLSpark is used to [help endangered species].

-   Explore generative adversarial artwork in [our collaboration with The MET and MIT].

-   Explore [our collaboration with Apache Spark] on image analysis.

[website]: https://mmlspark.blob.core.windows.net/website/index.html "aka.ms/spark"

[the Spark+AI Summit 2018]: https://databricks.com/sparkaisummit/north-america/spark-summit-2018-keynotes#Intelligent-cloud "Developing for the Intelligent Cloud and Intelligent Edge"

[the Spark+AI Summit 2019]: https://youtu.be/T_fs4C0aqD0?t=425

[the Spark+AI European Summit 2018]: https://youtu.be/N3ozCZXeOeU?t=472

[our paper]: https://arxiv.org/abs/1804.04031 "Flexible and Scalable Deep Learning with MMLSpark"

[help endangered species]: https://www.microsoft.com/en-us/ai/ai-lab-stories?activetab=pivot1:primaryr3 "Identifying snow leopards with AI"

[our collaboration with The MET and MIT]: https://www.microsoft.com/en-us/ai/ai-lab-stories?activetab=pivot1:primaryr4 "Generative art at the MET"

[our collaboration with Apache Spark]: https://blogs.technet.microsoft.com/machinelearning/2018/03/05/image-data-support-in-apache-spark/ "Image Data Support in Apache Spark"

[MMLSpark in Azure Machine Learning]: https://docs.microsoft.com/en-us/azure/machine-learning/preview/how-to-use-mmlspark "How to Use Microsoft Machine Learning Library for Apache Spark"

[MMLSpark at the Spark Summit]: https://databricks.com/session/mmlspark-lessons-from-building-a-sparkml-compatible-machine-learning-library-for-apache-spark "MMLSpark: Lessons from Building a SparkML-Compatible Machine Learning Library for Apache Spark"

## Contributing & feedback

This project has adopted the [Microsoft Open Source Code of Conduct].  For more
information see the [Code of Conduct FAQ] or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional
questions or comments.

[Microsoft Open Source Code of Conduct]: https://opensource.microsoft.com/codeofconduct/

[Code of Conduct FAQ]: https://opensource.microsoft.com/codeofconduct/faq/

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

To give feedback and/or report an issue, open a [GitHub
Issue](https://help.github.com/articles/creating-an-issue/).

## Other relevant projects
-   [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit)

-   [LightGBM](https://github.com/Microsoft/LightGBM)

-   [DMTK: Microsoft Distributed Machine Learning Toolkit](https://github.com/Microsoft/DMTK)

-   [Recommenders](https://github.com/Microsoft/Recommenders)

-   [JPMML-SparkML plugin for converting MMLSpark LightGBM models to
    PMML](https://github.com/alipay/jpmml-sparkml-lightgbm)

-   [Microsoft Cognitive Toolkit](https://github.com/Microsoft/CNTK)

-   [Azure Machine Learning
    preview features](https://docs.microsoft.com/en-us/azure/machine-learning/preview)

_Apache®, Apache Spark, and Spark® are either registered trademarks or
trademarks of the Apache Software Foundation in the United States and/or other
countries._
