# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys

if sys.version >= '3':
    basestring = str

import pyspark
from pyspark.ml.common import inherit_doc
from pyspark.sql.types import *
from pyspark.sql.types import Row, _create_row
import numpy as np
from mmlspark.opencv._ImageTransformer import _ImageTransformer

ImageFields = ["path", "height", "width", "type", "bytes"]

ImageSchema = StructType([
    StructField(ImageFields[0], StringType(),  True),
    StructField(ImageFields[1], IntegerType(), True),
    StructField(ImageFields[2], IntegerType(), True),
    StructField(ImageFields[3], IntegerType(), True),                 # OpenCV type: CV_8U in most cases
    StructField(ImageFields[4], BinaryType(), True) ])   # OpenCV bytes: row-wise BGR in most cases

def toNDArray(image):
    """
    Converts an image to a 1-dimensional array

    Args:
        image (object): The image to be converted

    Returns:
        array: The image as a 1-dimensional array
    """
    return np.asarray(image.bytes, dtype = np.uint8).reshape((image.height, image.width, 3))[:,:,(2,1,0)]

def toImage(array, path = "", ocvType = 16):
    """

    Converts a one-dimensional array to a 2 dimensional image

    Args:
        array (array):
        path (str):
        ocvType (int):

    Returns:
        object: 2 dimensional image
    """
    length = np.prod(array.shape)

    data = bytearray(array.astype(dtype=np.int8)[:,:,(2,1,0)].reshape(length))
    height = array.shape[0]
    width = array.shape[1]
    # Creating new Row with _create_row(), because Row(name = value, ... ) orders fields by name,
    # which conflicts with expected ImageSchema order when the new DataFrame is created by UDF
    return  _create_row(ImageFields, [path, height, width, ocvType, data])

from pyspark.ml.common import inherit_doc
@inherit_doc
class ImageTransformer(_ImageTransformer):
    """
    Resizes the image to the given width and height

    Args:
        height (int): The height to resize to (>=0)
        width (int): The width to resize to (>=0)

    """

    def resize(self, height, width):
        """
       Resizes the image to the given width and height

       Args:
           height (int): The height to resize to (>=0)
           width (int): The width to resize to (>=0)

       """
        self._java_obj.resize(height, width)
        return self

    def crop(self, x, y, height, width):
        """
        Crops the image given the starting x,y coordinates
        and the width and height

        Args:
            x (int): The initial x coordinate (>=0)
            y (int): The initial y coordinate (>=0)
            height (int): The height to crop to (>=0)
            width (int): The width to crop to (>=0)

        """
        self._java_obj.crop(x,y,height,width)
        return self

    def colorFormat(self, format):
        """
        Formats the image to the given image format

        Args:
            format (int): The format to convert to, please see OpenCV cvtColor function documentation for all formats

        """
        self._java_obj.colorFormat(format)
        return self

    def blur(self, height, width):
        """
        Blurs the image using a normalized box filter

        Args:
            height (double): The height of the box filter (>= 0)
            width (double): The width of the box filter (>= 0)

        """
        self._java_obj.blur(height, width)
        return self

    def threshold(self, threshold, maxVal, thresholdType):
        """
        Thresholds the image, please see OpenCV threshold function documentation for more information

        Args:
            threshold: (double) The threshold value
            maxVal (double): The maximum value to use
            thresholdType (double): The type of threshold, can be binary, binary_inv, trunc, zero, zero_inv

        """
        self._java_obj.threshold(threshold, maxVal, thresholdType)
        return self

    def gaussianKernel(self, appertureSize, sigma):
        """
        Blurs the image by applying a gaussian kernel

        Args:
            appertureSize (double): The aperture size, which should be odd and positive
            sigma (double): The standard deviation of the gaussian

        """
        self._java_obj.gaussianKernel(appertureSize, sigma)
        return self

    """
    Flips the image
    :param int flipCode: a flag to specify how to flip the image
     - 0 means flipping around the x-axis (up-down)
     - positive value (for example, 1) means flipping around y-axis (left-right, default)
     - negative value (for example, -1) means flipping around both axes (diagonally)
    See OpenCV documentation for details.
    """
    def flip(self, flipCode = 1):
        self._java_obj.flip(flipCode)
        return self
