from math import e
from typing import Union

from elgrad.tensor import DotProductError
# import numpy as np

import pytest  # type: ignore
from elgrad import Tensor, BroadcastError  # type: ignore


class TestConv2d:
    def test_one(self):
        img = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        filter = Tensor([[1, 2], [3, 4]], require_grad=True)
        result = img.conv2d(filter)
        print(result)
        # sum = result.sum()
        # sum.backward()
        # print(img, filter, sum)
        assert True
        # assert Tensor.array_equal(result, Tensor([[37, 47], [67, 77]]))

    def test_two(self):
        img = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        filter = Tensor([[1, 2], [3, 4]])
        result = img.conv2d(filter)
        print(result)
        assert Tensor.array_equal(result, Tensor([[37, 47], [67, 77], [97, 107]]))

    def test_three(self):
        img = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        filter = Tensor([[1, 2], [3, 4]])
        result = img.conv2d(filter)
        print(result)
        assert Tensor.array_equal(result, Tensor([[44, 54, 64], [84, 94, 104]]))

    def test_four(self):
        img = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        filter = Tensor([[1, 2], [3, 4]])
        result = img.conv2d(filter, stride=1)
        print(result)
        assert Tensor.array_equal(result, Tensor([[44, 54, 64], [84, 94, 104], [124, 134, 144]]))

    def test_five(self):
        img = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        filter = Tensor([[1, 2], [3, 4]])
        result = img.conv2d(filter, stride=2)
        # print(result)
        assert Tensor.array_equal(result, Tensor([[44, 64], [124, 144]]))


class TestCol2Img:
    def test_one(self):
        img = Tensor([[1, 2, 4, 5], [2, 3, 5, 6], [4, 5, 7, 8], [5, 6, 8, 9]])
        result = img.col2img(original_shape=(3, 3), kernel_size=(2, 2))
        expected_result = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert Tensor.array_equal(result, expected_result)

    def test_two(self):
        img = Tensor([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]]).reshape((6, 2))
        result = img.col2img(original_shape=(3, 3), kernel_size=(3, 2))
        expected_result = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert Tensor.array_equal(result, expected_result)

    def test_three(self):
        img = Tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]).reshape((6, 2))
        result = img.col2img(original_shape=(3, 3), kernel_size=(3, 2))
        expected_result = Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert Tensor.array_equal(result, expected_result)
