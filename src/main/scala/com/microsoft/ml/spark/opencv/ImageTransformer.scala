// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.opencv

import com.microsoft.ml.spark.codegen.Wrappable
import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol}
import com.microsoft.ml.spark.core.schema.{BinaryFileSchema, ImageSchemaUtils}
import com.microsoft.ml.spark.logging.BasicLogging
import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.ml.param.{ParamMap, _}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{ImageInjections, Transformer}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.opencv.core.{Core, Mat, Rect, Size}
import org.opencv.imgproc.Imgproc

import scala.collection.mutable.ListBuffer

//scalastyle:off field.name
/** Image processing stage.
  *
  * @param params Map of parameters
  */
abstract class ImageTransformerStage(params: Map[String, Any]) extends Serializable {
  def apply(image: Mat): Mat

  val stageName: String
}

/** Resizes the image. The parameters of the ParameterMap are:
  * "height" - the height of the image
  * "width"
  * "stageName"
  * Please refer to [[http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize OpenCV]]
  * for more information
  *
  * @param params ParameterMap of the parameters
  */
class ResizeImage(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val height: Double = params(ResizeImage.height).asInstanceOf[Int].toDouble
  val width: Double = params(ResizeImage.width).asInstanceOf[Int].toDouble
  override val stageName: String = ResizeImage.stageName

  override def apply(image: Mat): Mat = {
    val resized = new Mat()
    val sz = new Size(width, height)
    Imgproc.resize(image, resized, sz)
    resized
  }
}

/** Resize object contains the information for resizing;
  * "height"
  * "width"
  * "stageName" = "resize"
  */
object ResizeImage {
  val stageName = "resize"
  val height = "height"
  val width = "width"
}

/** Crops the image for processing. The parameters are:
  * "x" - First dimension; start of crop
  * "y" - second dimension - start of crop
  * "height" -height of cropped image
  * "width" - width of cropped image
  * "stageName" - "crop"
  *
  * @param params ParameterMap of the dimensions for cropping
  */
class CropImage(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val x: Int = params(CropImage.x).asInstanceOf[Int]
  val y: Int = params(CropImage.y).asInstanceOf[Int]
  val height: Int = params(CropImage.height).asInstanceOf[Int]
  val width: Int = params(CropImage.width).asInstanceOf[Int]
  override val stageName: String = CropImage.stageName

  override def apply(image: Mat): Mat = {
    val rect = new Rect(x, y, width, height)
    new Mat(image, rect)
  }
}

object CropImage {
  val stageName = "crop"
  val x = "x"
  val y = "y"
  val height = "height"
  val width = "width"
}

/** Converts an image from one color space to another, eg COLOR_BGR2GRAY. Refer to
  * [[http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor OpenCV]]
  * for more information.
  *
  * @param params Map of parameters and values
  */
class ColorFormat(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val format: Int = params(ColorFormat.format).asInstanceOf[Int]
  override val stageName: String = ColorFormat.stageName

  override def apply(image: Mat): Mat = {
    val dst = new Mat()
    Imgproc.cvtColor(image, dst, format)
    dst
  }
}

object ColorFormat {
  val stageName = "colorformat"
  val format = "format"
}

/** Flips the image
  *
  * @param params
  */
class Flip(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val flipCode: Int = params(Flip.flipCode).asInstanceOf[Int]
  override val stageName: String = Flip.stageName

  override def apply(image: Mat): Mat = {
    val dst = new Mat()
    Core.flip(image, dst, flipCode)
    dst
  }
}

object Flip {
  val stageName: String = "flip"
  val flipCode: String = "flipCode"

  val flipUpDown: Int = 0
  val flipLeftRight: Int = 1
  val flipBoth: Int = -1
}

/** Blurs the image using a box filter.
  * The com.microsoft.ml.spark.core.serialize.params are a map of the dimensions of the blurring box. Please refer to
  * [[http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#blur OpenCV]] for more information.
  *
  * @param params
  */
class Blur(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val height: Double = params(Blur.height).asInstanceOf[Double]
  val width: Double = params(Blur.width).asInstanceOf[Double]
  override val stageName: String = Blur.stageName

  override def apply(image: Mat): Mat = {
    val dst = new Mat()
    Imgproc.blur(image, dst, new Size(height, width))
    dst
  }
}

object Blur {
  val stageName: String = "blur"
  val height: String = "height"
  val width: String = "width"
}

/** Applies a threshold to each element of the image. Please refer to
  * [[http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#threshold threshold]] for
  * more information
  *
  * @param params
  */
class Threshold(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val threshold: Double = params(Threshold.threshold).asInstanceOf[Double]
  val maxVal: Double = params(Threshold.maxVal).asInstanceOf[Double]
  // EG Imgproc.THRESH_BINARY
  val thresholdType: Int = params(Threshold.thresholdType).asInstanceOf[Int]
  override val stageName: String = Threshold.stageName

  override def apply(image: Mat): Mat = {
    val dst = new Mat()
    Imgproc.threshold(image, dst, threshold, maxVal, thresholdType)
    dst
  }
}

object Threshold {
  val stageName: String = "threshold"
  val threshold: String = "threshold"
  val maxVal: String = "maxVal"
  val thresholdType: String = "type"
}

/** Applies gaussian kernel to blur the image. Please refer to
  * [[http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#gaussianblur OpenCV]] for detailed information
  * about the parameters and their allowable values.
  *
  * @param params Map of parameter values containg the aperture and sigma for the kernel.
  */
class GaussianKernel(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val appertureSize: Int = params(GaussianKernel.apertureSize).asInstanceOf[Int]
  val sigma: Double = params(GaussianKernel.sigma) match {
    case d: Double => d
    case i: Int => i.toDouble
  }
  override val stageName: String = GaussianKernel.stageName

  override def apply(image: Mat): Mat = {
    val dst = new Mat()
    val kernel = Imgproc.getGaussianKernel(appertureSize, sigma)
    Imgproc.filter2D(image, dst, -1, kernel)
    dst
  }
}

object GaussianKernel {
  val stageName: String = "gaussiankernel"
  val apertureSize: String = "apertureSize"
  val sigma: String = "sigma"
}

/** Pipelined image processing. */
object ImageTransformer extends DefaultParamsReadable[ImageTransformer] {

  override def load(path: String): ImageTransformer = super.load(path)

  /** Convert Spark image representation to OpenCV format. */
  private def row2mat(row: Row): (String, Mat) = {
    val path = ImageSchema.getOrigin(row)
    val height = ImageSchema.getHeight(row)
    val width = ImageSchema.getWidth(row)
    val ocvType = ImageSchema.getMode(row)
    val bytes = ImageSchema.getData(row)

    val img = new Mat(height, width, ocvType)
    img.put(0, 0, bytes)
    (path, img)
  }

  /** Convert from OpenCV format to Dataframe Row; unroll if needed. */
  private def mat2row(img: Mat, path: String = ""): Row = {
    val ocvBytes = new Array[Byte](img.total.toInt * img.elemSize.toInt)
    img.get(0, 0, ocvBytes) //extract OpenCV bytes
    Row(path, img.height, img.width, img.channels(), img.`type`, ocvBytes)
  }

  /** Apply all OpenCV transformation stages to a single image; unroll the result if needed
    * For null inputs or binary files that could not be parsed, return None.
    * Break on OpenCV errors.
    */
  def process(stages: Seq[ImageTransformerStage], decodeMode: String)(r: Any): Option[Row] = {

    if (r == null) return None

    val decoded = (r, decodeMode) match {
      case (row: Row, "binaryfile") =>
        val path = BinaryFileSchema.getPath(row)
        val bytes = BinaryFileSchema.getBytes(row)

        //early return if the image can't be decompressed
        ImageInjections.decode(path, bytes).getOrElse(return None).getStruct(0)
      case (bytes: Array[Byte], "binary") =>
        ImageInjections.decode(null, bytes).getOrElse(return None).getStruct(0)
      case (row: Row, "image") =>
        row
      case (_, mode) =>
        throw new MatchError(s"Unknown decoder mode $mode")
    }

    val (path, img) = row2mat(decoded)
    val result = stages.foldLeft(img) {
      case (imgInternal, stage) => stage.apply(imgInternal)
    }
    Some(mat2row(result, path))
  }

}

/** Image processing stage. Please refer to OpenCV for additional information
  *
  * @param uid The id of the module
  */
class ImageTransformer(val uid: String) extends Transformer
  with HasInputCol with HasOutputCol with Wrappable with DefaultParamsWritable with BasicLogging {
  logClass()

  import ImageTransformer._

  override protected lazy val pyInternalWrapper = true

  def this() = this(Identifiable.randomUID("ImageTransformer"))

  val stages: ArrayMapParam = new ArrayMapParam(this, "stages", "Image transformation stages")

  def setStages(value: Array[Map[String, Any]]): this.type = set(stages, value)

  val emptyStages: Array[Map[String, Any]] = Array[Map[String, Any]]()

  def getStages: Array[Map[String, Any]] = if (isDefined(stages)) $(stages) else emptyStages

  private def addStage(stage: Map[String, Any]): this.type = set(stages, getStages :+ stage)

  setDefault(inputCol -> "image", outputCol -> (uid + "_output"))

  // every stage has a name like "resize", "normalize", "unroll"
  val stageName = "action"

  def resize(height: Int, width: Int): this.type = {
    require(width >= 0 && height >= 0, "width and height should be nonnegative")

    addStage(Map(stageName -> ResizeImage.stageName,
      ResizeImage.width -> width,
      ResizeImage.height -> height))
  }

  def crop(x: Int, y: Int, height: Int, width: Int): this.type = {
    require(x >= 0 && y >= 0 && width >= 0 && height >= 0, "crop values should be nonnegative")

    addStage(Map(stageName -> CropImage.stageName,
      CropImage.width -> width,
      CropImage.height -> height,
      CropImage.x -> x,
      CropImage.y -> y))
  }

  def colorFormat(format: Int): this.type = {
    addStage(Map(stageName -> ColorFormat.stageName, ColorFormat.format -> format))
  }

  def blur(height: Double, width: Double): this.type = {
    addStage(Map(stageName -> Blur.stageName, Blur.height -> height, Blur.width -> width))
  }

  def threshold(threshold: Double, maxVal: Double, thresholdType: Int): this.type = {
    addStage(Map(stageName -> Threshold.stageName,
      Threshold.maxVal -> maxVal,
      Threshold.threshold -> threshold,
      Threshold.thresholdType -> thresholdType))
  }

  /** Flips the image
    *
    * @param flipCode is a flag to specify how to flip the image:
    * - 0 means flipping around the x-axis (i.e. up-down)
    * - positive value (for example, 1) means flipping around y-axis (left-right)
    * - negative value (for example, -1) means flipping around both axes (diagonally)
    *                 See OpenCV documentation for details.
    * @return
    */
  def flip(flipCode: Int): this.type = {
    addStage(Map(stageName -> Flip.stageName, Flip.flipCode -> flipCode))
  }

  def gaussianKernel(apertureSize: Int, sigma: Double): this.type = {
    addStage(Map(stageName -> GaussianKernel.stageName,
      GaussianKernel.apertureSize -> apertureSize,
      GaussianKernel.sigma -> sigma))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({

      //  load native OpenCV library on each partition
      // TODO: figure out more elegant way
      val df = OpenCVUtils.loadOpenCV(dataset.toDF)
      val decodeMode = df.schema(getInputCol).dataType match {
        case s if ImageSchemaUtils.isImage(s) => "image"
        case s if BinaryFileSchema.isBinaryFile(s) => "binaryfile"
        case s if s == BinaryType => "binary"
        case s =>
          throw new IllegalArgumentException(s"input column should have Image or BinaryFile type, got $s")

      }

      val transforms = ListBuffer[ImageTransformerStage]()
      for (stage <- getStages) {
        stage(stageName) match {
          case ResizeImage.stageName => transforms += new ResizeImage(stage)
          case CropImage.stageName => transforms += new CropImage(stage)
          case ColorFormat.stageName => transforms += new ColorFormat(stage)
          case Blur.stageName => transforms += new Blur(stage)
          case Threshold.stageName => transforms += new Threshold(stage)
          case GaussianKernel.stageName => transforms += new GaussianKernel(stage)
          case Flip.stageName => transforms += new Flip(stage)
          case unsupported: String => throw new IllegalArgumentException(s"unsupported transformation $unsupported")
        }
      }

      val convert = UDFUtils.oldUdf(process(transforms, decodeMode = decodeMode) _, ImageSchema.columnSchema)

      df.withColumn(getOutputCol, convert(df(getInputCol)))
    })
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add(getOutputCol, ImageSchema.columnSchema)
  }

}

//scalastyle:on field.name
