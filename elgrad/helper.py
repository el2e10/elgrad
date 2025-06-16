from math import floor
from typing import Union, Tuple

import numpy as np


def check_conv2d_stride_shape(inputr, filter, stride: Union[int, Tuple[int]]) -> Tuple[bool, str]:
    if not isinstance(stride, int | tuple):
        return False, "Stride should be an int or tuple"
    if isinstance(stride, int) and (stride > min(filter.shape[-2:])):
        return False, "Invalid stride length"
    if isinstance(stride, tuple) and (len(stride) != 2):
        return False, "Invalid stride dimension"
    return True, ""


def get_indices_for_img_col_transformation(input_shape: tuple, kernel_shape: tuple, stride, padding):
    stride = tuple([stride, stride]) if isinstance(stride, int) else stride
    _, i_c, i_h, i_w = input_shape
    _, _, k_h, k_w = kernel_shape

    o_h, o_w = (
        floor((i_h + (2 * padding) - k_h) / stride[0] + 1),
        floor((i_w + (2 * padding) - k_w) / stride[1] + 1),
    )

    # Getting i
    first = np.arange(k_h)
    level_1 = np.tile(np.repeat(first, k_w), i_c)
    every_level = stride[0] * np.repeat(np.arange(o_h), o_w)
    i = level_1.reshape(-1, 1) + every_level.reshape((1, -1))

    # Getting j
    slide_1 = np.tile(np.tile(np.arange(k_w), k_h), i_c)
    every_slide = stride[1] * np.tile(np.arange(o_w), o_h)
    j = slide_1.reshape(-1, 1) + every_slide.reshape(1, -1)

    # # Getting c
    c = np.repeat(np.arange(i_c), k_h * k_w).reshape(-1, 1)

    return c, i, j


if __name__ == "__main__":
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]).reshape((1, 2, 3, 3))
    kernel = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]).reshape((1, 2, 2, 2))

    c, i, j = get_indices_for_img_col_transformation(img.shape, kernel.shape, 1, 0)
    print(i, j, c)
    location = img[:, c, i, j]
    # print(i)
