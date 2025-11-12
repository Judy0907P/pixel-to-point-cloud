# %% [markdown]
# # Triangulation
#
# This function triangulates 3D points from two sets of two undistorted normalized
# pixels and a [`TransformationMatrix`](transformation_matrix.py) object. The process
# for this was discussed in more detail in the workshop
# [5: Dual Camera Setups](../workshops/05_dual_camera_setups.ipynb).


# %%
import numpy as np
from nptyping import Float32, NDArray, Shape

from oaf_vision_3d.lens_model import LensModel
from oaf_vision_3d.transformation_matrix import TransformationMatrix


def triangulate_points(  # type: ignore
    undistorted_normalized_pixels_0: NDArray[Shape["H, W, 2"], Float32],
    undistorted_normalized_pixels_1: NDArray[Shape["H, W, 2"], Float32],
    transformation_matrix: TransformationMatrix,
) -> NDArray[Shape["H, W, 3"], Float32]:
    H = undistorted_normalized_pixels_0.shape[0]
    W = undistorted_normalized_pixels_0.shape[1]
    ones = np.ones(shape=(H, W, 1), dtype=undistorted_normalized_pixels_0.dtype)
    v0 = np.concatenate([undistorted_normalized_pixels_0, ones], axis=-1)
    P1 = transformation_matrix.translation
    R = transformation_matrix.rotation.as_matrix()
    u1 = np.concatenate([undistorted_normalized_pixels_1, ones], axis=-1)
    v1 = np.tensordot(u1, R.T, axes=1)
    a = np.sum(v0 * v0, axis=-1)
    b = np.sum(v0 * v1, axis=-1)
    c = np.sum(v1 * v1, axis=-1)
    d = np.tensordot(v0, P1, axes=([2], [0]))
    e = np.tensordot(v1, P1, axes=([2], [0]))
    t = (b * e - c * d) / (b * b - a * c)
    P = t[..., np.newaxis] * v0
    return P


def triangulate_disparity(
    disparity: NDArray[Shape["H, W"], Float32],
    lens_model_0: LensModel,
    lens_model_1: LensModel,
    transformation_matrix: TransformationMatrix,
) -> NDArray[Shape["H, W, 3"], Float32]:
    y, x = np.indices(disparity.shape, dtype=np.float32)
    pixels_0 = np.stack([x, y], axis=-1)
    pixels_1 = np.stack([x - disparity, y], axis=-1)

    undistortied_normalized_pixels_0 = lens_model_0.undistort_pixels(
        normalized_pixels=lens_model_0.normalize_pixels(pixels=pixels_0)
    )
    undistortied_normalized_pixels_1 = lens_model_1.undistort_pixels(
        normalized_pixels=lens_model_1.normalize_pixels(pixels=pixels_1)
    )

    return triangulate_points(
        undistorted_normalized_pixels_0=undistortied_normalized_pixels_0,
        undistorted_normalized_pixels_1=undistortied_normalized_pixels_1,
        transformation_matrix=transformation_matrix,
    )
