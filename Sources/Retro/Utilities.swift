// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import TensorFlow

/// Converts one or more images from RGB to Grayscale. Returns a tensor with the same rank as
/// `images`. The size of the last dimension of the result is `1` and it contains the Grayscale
/// value of the pixels.
///
/// - Parameter images: Tensor containing RGB image(s). Its last dimension must have size 3.
/// - Returns: Tensor containing the converted Grayscale image(s).
@differentiable
@inlinable
public func convertRGBToGrayscale<Scalar: TensorFlowFloatingPoint>(
  _ images: Tensor<Scalar>
) -> Tensor<Scalar> {
  let rgbWeights = Tensor<Scalar>([0.2989, 0.5870, 0.1140])
  return (images * rgbWeights).sum(alongAxes: -1)
}

@usableFromInline
internal enum ImageResizeMethod {
  case nearestNeighbor
  case bilinear
  case bicubic
  case area

  @inlinable
  internal func resize(
    images: Tensor<Float>,
    size: Tensor<Int32>,
    alignCorners: Bool = false
  ) -> Tensor<Float> {
    switch self {
    case .nearestNeighbor:
      return Raw.resizeNearestNeighbor(images: images, size: size, alignCorners: alignCorners)
    case .bilinear:
      return Raw.resizeBilinear(images: images, size: size, alignCorners: alignCorners)
    case .bicubic:
      return Raw.resizeBicubic(images: images, size: size, alignCorners: alignCorners)
    case .area:
      return Raw.resizeArea(images: images, size: size, alignCorners: alignCorners)
    }
  }
}

@inlinable
internal func resize<Scalar: TensorFlowNumeric>(
  images: Tensor<Scalar>,
  to size: Tensor<Int32>,
  method: ImageResizeMethod = .bilinear,
  alignCorners: Bool = false,
  preserveAspectRatio: Bool = false
) -> Tensor<Scalar> {
  precondition(images.rank == 3 || images.rank == 4, "'images' must be of rank 3 or 4.")
  precondition(size.rank == 1 && size.shape[0] == 2, "'size' must be a vector with 2 elements.")
  let batched = images.rank == 4
  let images = batched ? images : images.expandingShape(at: 0)
  let height = images.shape[1]
  let width = images.shape[2]
  var newHeight = size[0].scalarized()
  var newWidth = size[1].scalarized()

  // Compute appropriate new size based on whether `preserveAspectRatio` is `true`.
  if preserveAspectRatio {
    let heightScaleFactor = Float(newHeight) / Float(height)
    let widthScaleFactor = Float(newWidth) / Float(width)
    let scaleFactor = min(heightScaleFactor, widthScaleFactor)
    newHeight = Int32(scaleFactor * Float(height))
    newWidth = Int32(scaleFactor * Float(width))
  }

  // Check if the resize is necessary.
  if height == newHeight && width == newWidth {
    return batched ? images : images.squeezingShape(at: 0)
  }

  let resizedImages = Tensor<Scalar>(method.resize(
    images: Tensor<Float>(images),
    size: preserveAspectRatio ? Tensor<Int32>([Int32(newHeight), Int32(newWidth)]) : size,
    alignCorners: alignCorners))
  
  return batched ? resizedImages : resizedImages.squeezingShape(at: 0)
}
