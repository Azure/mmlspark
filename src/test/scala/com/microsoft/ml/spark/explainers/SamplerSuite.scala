package com.microsoft.ml.spark.explainers

import breeze.stats.distributions.RandBasis
import com.microsoft.ml.spark.core.test.base.TestBase
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.abs
import breeze.stats.{mean, stddev}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

class SamplerSuite extends TestBase {
  test("ContinuousFeatureStats can draw samples") {
    implicit val randBasis: RandBasis = RandBasis.withSeed(123)

    val featureStats = ContinuousFeatureStats(1.5, DoubleType)
    val samples = BDV((1 to 1000) map {
      _ => featureStats.sample(3.0)
    }: _*)

    // println(samples(0 to 100))

    assert(abs(mean(samples) - 2.9557393788483997) < 1e-5)
    assert(abs(stddev(samples) - 1.483087711702025) < 1e-5)
  }

  test("DiscreteFeatureStats can draw samples") {
    implicit val randBasis: RandBasis = RandBasis.withSeed(123)

    val freqTable = Map(2d -> 60d, 1d -> 900d, 3d -> 40d)

    val featureStats = DiscreteFeatureStats(freqTable, DoubleType)
    val samples = BDV((1 to 1000) map {
      _ => featureStats.sample(3.0)
    }: _*)

    // println(samples(0 to 100))

    assert(samples.findAll(_ == 1.0).size == 896)
    assert(samples.findAll(_ == 2.0).size == 64)
    assert(samples.findAll(_ == 3.0).size == 40)
  }

  test("LIMEVectorSampler can draw samples") {
    implicit val randBasis: RandBasis = RandBasis.withSeed(123)
    val featureStats = Seq(
      ContinuousFeatureStats(5.3, DoubleType),
      DiscreteFeatureStats(Map(2d -> 60d, 1d -> 900d, 3d -> 40d), DoubleType)
    )

    val sampler = new LIMEVectorSampler(featureStats)
    val samples = (1 to 1000).map {
      _ => sampler.sample(BDV(3.2, 1.0))
    }

    val sampleMatrix = BDM(samples: _*)

//    println(mean(sampleMatrix(::, 0)))
//    println(stddev(sampleMatrix(::, 0)))
//
//    println(sampleMatrix(::, 1).findAll(_ == 1d).size)
//    println(sampleMatrix(::, 1).findAll(_ == 2d).size)
//    println(sampleMatrix(::, 1).findAll(_ == 3d).size)

    assert(abs(mean(sampleMatrix(::, 0)) - 2.9636538120292903) < 1e-5)
    assert(abs(stddev(sampleMatrix(::, 0)) - 5.3043309761267565) < 1e-5)

    assert(sampleMatrix(::, 1).findAll(_ == 1d).size == 886)
    assert(sampleMatrix(::, 1).findAll(_ == 2d).size == 68)
    assert(sampleMatrix(::, 1).findAll(_ == 3d).size == 46)
  }

  test("LIMETabularSampler can draw samples") {
    implicit val randBasis: RandBasis = RandBasis.withSeed(123)
    val featureStats = Seq(
      ContinuousFeatureStats(5.3, DoubleType),
      DiscreteFeatureStats(Map(2d -> 60d, 1d -> 900d, 3d -> 40d), IntegerType)
    )

    val sampler = new LIMETabularSampler(featureStats)

    val row = Row.fromSeq(Array[Any](3.2d, 1))

    val samples = (1 to 1000).map {
      _ =>
        val sample = sampler.sample(row)
        BDV(sample.getAs[Double](0), sample.getAs[Int](1).toDouble)
    }

    val sampleMatrix = BDM(samples: _*)

//    println(mean(sampleMatrix(::, 0)))
//    println(stddev(sampleMatrix(::, 0)))
//
//    println(sampleMatrix(::, 1).findAll(_ == 1d).size)
//    println(sampleMatrix(::, 1).findAll(_ == 2d).size)
//    println(sampleMatrix(::, 1).findAll(_ == 3d).size)

    assert(abs(mean(sampleMatrix(::, 0)) - 2.9636538120292903) < 1e-5)
    assert(abs(stddev(sampleMatrix(::, 0)) - 5.3043309761267565) < 1e-5)

    assert(sampleMatrix(::, 1).findAll(_ == 1d).size == 886)
    assert(sampleMatrix(::, 1).findAll(_ == 2d).size == 68)
    assert(sampleMatrix(::, 1).findAll(_ == 3d).size == 46)
  }
}
