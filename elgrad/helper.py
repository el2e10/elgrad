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

    stride = tuple([stride, stride, stride, stride]) if isinstance(stride, int) else stride
    i_b, i_c, i_h, i_w = input_shape
    k_b, k_c, k_h, k_w = kernel_shape

    o_b, o_c, o_h, o_w = (
        floor((i_b + (2 * padding) - k_b) / stride[0] + 1),
        floor((i_c + (2 * padding) - k_c) / stride[1] + 1),
        floor((i_h + (2 * padding) - k_h) / stride[2] + 1),
        floor((i_w + (2 * padding) - k_w) / stride[3] + 1),
    )
    print("Output shape is ", o_b, o_c, o_h, o_w)

    # Getting i
    first = np.arange(k_h)
    level_1 = np.repeat(first, k_w * k_b * k_c)
    levels = np.tile(np.repeat(np.arange(o_h), o_w), o_b * o_c).reshape(-1, 1)
    h = level_1 + levels

    # Getting j
    first = np.arange(k_w)
    level_1 = np.tile(np.repeat(first, k_b * k_c), k_h)
    levels = np.tile(np.arange(o_w), o_b * o_c * o_h).reshape(-1, 1)
    w = level_1 + levels
    # print(w)

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
    # print(b)

    return b, c, h, w


if __name__ == "__main__":
    img = np.arange(64).reshape((2, 2, 4, 4))
    kernel_shape = (1, 1, 1, 1)

    i_b, i_c, i_h, i_w = img.shape
    k_b, k_c, k_h, k_w = kernel_shape

    o_b, o_c, o_h, o_w = (
        floor((i_b + (2 * 0) - k_b) / 1 + 1),
        floor((i_c + (2 * 0) - k_c) / 1 + 1),
        floor((i_h + (2 * 0) - k_h) / 1 + 1),
        floor((i_w + (2 * 0) - k_w) / 1 + 1),
    )

    # print(img)
    b, c, i, j = get_indices_for_img_col_transformation(img.shape, kernel_shape, 1, 0)
    location = img[b, c, i, j]
    # print(location)
    output_data = np.max(location, axis=-1, keepdims=True)
    output_data = output_data.reshape((o_b, o_c, o_h, o_w))
    print(output_data)
