""" This program is meant to serve as all-purpose utility script for this
    pipeline. Now, and probably in the future, it is filled with code lifted
    from scikit-image. I'm not sure whether it should be combined with nifti-util.
    It may occur that non-image functions may be needed across programs, and that
    this file could be a repository for them. However, the line is blurry between
    image and non-image.
"""

import numpy as np

def assert_nD(array, ndim, arg_name='image'):
    """
    Verify an array meets the desired ndims.
    Parameters
    ----------
    array : array-like
        Input array to be validated
    ndim : int or iterable of ints
        Allowable ndim or ndims for the array.
    arg_name : str, optional
        The name of the array in the original function.
    """
    array = np.asanyarray(array)
    msg = "The parameter `%s` must be a %s-dimensional array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if not array.ndim in ndim:
        raise ValueError(msg % (arg_name, '-or-'.join([str(n) for n in ndim])))