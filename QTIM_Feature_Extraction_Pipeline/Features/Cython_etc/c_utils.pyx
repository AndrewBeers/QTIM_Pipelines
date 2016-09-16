#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, abs
# from .._shared.interpolation cimport bilinear_interpolation, round
# from .._shared.transform cimport integrate
import cython

cdef extern from "numpy/npy_math.h":
    double NAN "NPY_NAN"

ctypedef fused any_int:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t

def _glcm_loop(any_int[:, ::1] image, double[:] distances,
               double[:] angles, Py_ssize_t levels,
               cnp.uint32_t[:, :, :, ::1] out):
    """Perform co-occurrence matrix accumulation.
    Parameters
    ----------
    image : ndarray
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    distances : ndarray
        List of pixel pair distance offsets.
    angles : ndarray
        List of pixel pair angles in radians.
    levels : int
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of grey-levels counted
        (typically 256 for an 8-bit image).
    out : ndarray
        On input a 4D array of zeros, and on output it contains
        the results of the GLCM computation.
    """

    cdef:
        Py_ssize_t a_idx, d_idx, r, c, rows, cols, row, col
        any_int i, j
        cnp.float64_t angle, distance

    # with nogil:
    rows = image.shape[0]
    cols = image.shape[1]

    for a_idx in range(angles.shape[0]):
        angle = angles[a_idx]
        for d_idx in range(distances.shape[0]):
            distance = distances[d_idx]
            for r in range(rows):
                for c in range(cols):
                    i = image[r, c]

                    # compute the location of the offset pixel
                    row = r + <int>round(sin(angle) * distance)
                    col = c + <int>round(cos(angle) * distance)

                    # make sure the offset is within bounds
                    if row >= 0 and row < rows and \
                       col >= 0 and col < cols:
                        j = image[row, col]

                        if i >= 0 and i < levels and \
                           j >= 0 and j < levels:
                            out[i, j, d_idx, a_idx] += 1