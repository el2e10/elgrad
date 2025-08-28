from math import floor
from typing import Union, Tuple

import numpy as np  # type: ignore


def check_conv2d_stride_shape(inputr, filter, stride: Union[int, Tuple[int]]) -> Tuple[bool, str]:
    if not isinstance(stride, int | tuple):
        return False, "Stride should be an int or tuple"
    if isinstance(stride, int) and (stride > min(filter.shape[-2:])):
        return False, "Invalid stride length"
    if isinstance(stride, tuple) and (len(stride) != 2):
        return False, "Invalid stride dimension"
    return True, ""


def get_indices_for_img_col_transformation(input_shape: tuple, kernel_shape: tuple, stride, padding):
    # TODO add input and kernel shape check
    # TODO add support for stride

    stride = tuple([stride, stride, stride, stride]) if isinstance(stride, int) else stride
    i_b, i_c, i_h, i_w = input_shape
    k_b, k_c, k_h, k_w = kernel_shape

    o_b, o_c, o_h, o_w = (
        floor((i_b + (2 * padding) - k_b) / stride[0] + 1),
        floor((i_c + (2 * padding) - k_c) / stride[1] + 1),
        floor((i_h + (2 * padding) - k_h) / stride[2] + 1),
        floor((i_w + (2 * padding) - k_w) / stride[3] + 1),
    )

    # Getting i
    first = np.arange(k_h)
    level_1 = np.tile(np.repeat(first, k_w), k_b * k_c)
    levels = np.tile(np.repeat(np.arange(o_h), o_w), o_b * o_c).reshape(-1, 1)
    h = level_1 + levels

    # Getting j
    first = np.arange(k_w)
    level_1 = np.tile(first, k_b * k_c * k_h)
    levels = np.tile(np.arange(o_w), o_b * o_c * o_h).reshape(-1, 1)
    w = level_1 + levels

    # Getting c
    first = np.arange(k_c)
    level_1 = np.tile(np.repeat(first, k_h * k_w), k_b)
    levels = np.tile(np.repeat(np.arange(o_c), o_h * o_w), o_b).reshape(-1, 1)
    c = level_1 + levels

    # Getting b
    first = np.arange(k_b)
    level_1 = np.tile(first, k_h * k_w * k_c)
    levels = np.repeat(np.arange(o_b), o_c * o_h * o_w).reshape(-1, 1)
    b = level_1 + levels

    return b, c, h, w


if __name__ == "__main__":
    pass
    # img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]).reshape(1, 2, 3, 3)
    #
    # kernel_shape = (1, 2, 2, 2)
    #
    # i_b, i_c, i_h, i_w = img.shape
    # k_b, k_c, k_h, k_w = kernel_shape
    #
    # o_b, o_c, o_h, o_w = (
    #     floor((i_b + (2 * 0) - k_b) / 1 + 1),
    #     floor((i_c + (2 * 0) - k_c) / 1 + 1),
    #     floor((i_h + (2 * 0) - k_h) / 1 + 1),
    #     floor((i_w + (2 * 0) - k_w) / 1 + 1),
    # )
    #
    # # print(img)
    # b, c, i, j = get_indices_for_img_col_transformation(img.shape, kernel_shape, 1, 0)
    # # print(b, c, i, j)
    # location = img[b, c, i, j]
    # print(location)
