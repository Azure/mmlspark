// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.opencv

import com.microsoft.ml.spark.core.contracts.{HasInputCol, HasOutputCol, MMLParams}
import com.microsoft.ml.spark.core.env.InternalWrapper
import com.microsoft.ml.spark.core.schema.{BinaryFileSchema, ImageSchema}
import com.microsoft.ml.spark.core.serialize.params.ArrayMapParam
import com.microsoft.ml.spark.IO.image.ImageReader
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.opencv.core.{Core, Mat, Rect, Size}
import org.opencv.imgproc.Imgproc

import scala.collection.mutable.ListBuffer

/** Image processing stage.
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
  * @param params ParameterMap of the parameters
  */
class ResizeImage(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val height = params(ResizeImage.height).asInstanceOf[Int].toDouble
  val width = params(ResizeImage.width).asInstanceOf[Int].toDouble
  override val stageName = ResizeImage.stageName

  override def apply(image: Mat): Mat = {
    var resized = new Mat()
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
  * @param params ParameterMap of the dimensions for cropping
  */
class CropImage(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val x = params(CropImage.x).asInstanceOf[Int]
  val y = params(CropImage.y).asInstanceOf[Int]
  val height = params(CropImage.height).asInstanceOf[Int]
  val width = params(CropImage.width).asInstanceOf[Int]
  override val stageName = CropImage.stageName

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
  * @param params Map of parameters and values
  */
class ColorFormat(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val format = params(ColorFormat.format).asInstanceOf[Int]
  override val stageName = ColorFormat.stageName

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
  * @param params
  */
class Flip(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val flipCode = params(Flip.flipCode).asInstanceOf[Int]
  override val stageName = Flip.stageName

  override def apply(image: Mat): Mat = {
    val dst = new Mat()
    Core.flip(image, dst, flipCode)
    dst
  }
}

object Flip {
  val stageName = "flip"
  val flipCode = "flipCode"

  val flipUpDown = 0
  val flipLeftRight = 1
  val flipBoth = -1
}

  /** Blurs the image using a box filter.
    * The params are a map of the dimensions of the blurring box. Please refer to
    * [[http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#blur OpenCV]] for more information.
    * @param params
    */
class Blur(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val height = params(Blur.height).asInstanceOf[Double]
  val width = params(Blur.width).asInstanceOf[Double]
  override val stageName = Blur.stageName

  override def apply(image: Mat): Mat = {
    val dst = new Mat()
    Imgproc.blur(image, dst, new Size(height, width))
    dst
  }
}

object Blur {
  val stageName = "blur"
  val height = "height"
  val width = "width"
}

/** Applies a threshold to each element of the image. Please refer to
  * [[http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#threshold threshold]] for
  * more information
  * @param params
  */
class Threshold(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val threshold = params(Threshold.threshold).asInstanceOf[Double]
  val maxVal = params(Threshold.maxVal).asInstanceOf[Double]
  // EG Imgproc.THRESH_BINARY
  val thresholdType = params(Threshold.thresholdType).asInstanceOf[Int]
  override val stageName = Threshold.stageName

  override def apply(image: Mat): Mat = {
    val dst = new Mat()
    Imgproc.threshold(image, dst, threshold, maxVal, thresholdType)
    dst
  }
}

object Threshold {
  val stageName = "threshold"
  val threshold = "threshold"
  val maxVal = "maxVal"
  val thresholdType = "type"
}

/** Applies gaussian kernel to blur the image. Please refer to
  * [[http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#gaussianblur OpenCV]] for detailed information
  * about the parameters and their allowable values.
  * @param params Map of parameter values containg the aperture and sigma for the kernel.
  */
class GaussianKernel(params: Map[String, Any]) extends ImageTransformerStage(params) {
  val appertureSize: Int = params(GaussianKernel.appertureSize).asInstanceOf[Int]
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
  val stageName = "gaussiankernel"
  val appertureSize = "appertureSize"
  val sigma = "sigma"
}

/** Pipelined image processing. */
object ImageTransformer extends DefaultParamsReadable[ImageTransformer] {

  override def load(path: String): ImageTransformer = super.load(path)

  /** Convert Spark image representation to OpenCV format. */
  private def row2mat(row: Row): (String, Mat) = {
    val path    = ImageSchema.getPath(row)
    val height  = ImageSchema.getHeight(row)
    val width   = ImageSchema.getWidth(row)
    val ocvType = ImageSchema.getType(row)
    val bytes   = ImageSchema.getBytes(row)

    val img = new Mat(height, width, ocvType)
    img.put(0,0,bytes)
    (path, img)
  }

  /**  Convert from OpenCV format to Dataframe Row; unroll if needed. */
  private def mat2row(img: Mat, path: String = ""): Row = {
    var ocvBytes = new Array[Byte](img.total.toInt*img.elemSize.toInt)
    img.get(0,0,ocvBytes)         //extract OpenCV bytes
    Row(path, img.height, img.width, img.`type`, ocvBytes)
  }

  /** Apply all OpenCV transformation stages to a single image; unroll the result if needed
    * For null inputs or binary files that could not be parsed, return None.
    * Break on OpenCV errors.
    */
  def process(stages: Seq[ImageTransformerStage], decode: Boolean)(row: Row): Option[Row] = {

    if (row == null) return None

    val decoded = if (decode) {
      val path  = BinaryFileSchema.getPath(row)
      val bytes = BinaryFileSchema.getBytes(row)

      //early return if the image can't be decompressed
      ImageReader.decode(path, bytes).getOrElse(return None)
    } else row

    var (path, img) = row2mat(decoded)
    for (stage <- stages) {
      img = stage.apply(img)
    }
    Some(mat2row(img, path))
  }

}

/** Image processing stage. Please refer to OpenCV for additional information
  * @param uid The id of the module
  */
@InternalWrapper
class ImageTransformer(val uid: String) extends Transformer
  with HasInputCol with HasOutputCol with MMLParams {

  import ImageTransformer._

  def this() = this(Identifiable.randomUID("ImageTransformer"))

  val stages: ArrayMapParam = new ArrayMapParam(this, "stages", "Image transformation stages")
  def setStages(value: Array[Map[String, Any]]): this.type = set(stages, value)
  val emptyStages = Array[Map[String, Any]]()
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
    * @param flipCode is a flag to specify how to flip the image:
    * - 0 means flipping around the x-axis (i.e. up-down)
    * - positive value (for example, 1) means flipping around y-axis (left-right)
    * - negative value (for example, -1) means flipping around both axes (diagonally)
    * See OpenCV documentation for details.
    * @return
    */
  def flip(flipCode: Int): this.type = {
    addStage(Map(stageName -> Flip.stageName, Flip.flipCode -> flipCode))
  }

  def gaussianKernel(appertureSize: Int, sigma: Double): this.type = {
    addStage(Map(stageName -> GaussianKernel.stageName,
                 GaussianKernel.appertureSize -> appertureSize,
                 GaussianKernel.sigma -> sigma))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {

    //  load native OpenCV library on each partition
    // TODO: figure out more elegant way
    val spark = dataset.sqlContext

    val schema = dataset.toDF.schema

    val df = ImageReader.loadOpenCV(dataset.toDF)

    val isBinary = BinaryFileSchema.isBinaryFile(df, $(inputCol))
    assert(ImageSchema.isImage(df, $(inputCol)) || isBinary,
           "input column should have Image or BinaryFile type")

    val transforms = ListBuffer[ImageTransformerStage]()
    for (stage <- getStages) {
      stage(stageName) match  {
        case ResizeImage.stageName    => transforms += new ResizeImage(stage)
        case CropImage.stageName      => transforms += new CropImage(stage)
        case ColorFormat.stageName    => transforms += new ColorFormat(stage)
        case Blur.stageName           => transforms += new Blur(stage)
        case Threshold.stageName      => transforms += new Threshold(stage)
        case GaussianKernel.stageName => transforms += new GaussianKernel(stage)
        case Flip.stageName           => transforms += new Flip(stage)
        case unsupported: String => throw new IllegalArgumentException(s"unsupported transformation $unsupported")
      }
    }

    val func = process(transforms, decode = isBinary)(_)
    val convert = udf(func, ImageSchema.columnSchema)

    df.withColumn($(outputCol), convert(df($(inputCol))))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add($(outputCol), ImageSchema.columnSchema)
  }

}
