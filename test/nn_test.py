from math import e
from typing import Union

from elgrad.tensor import DotProductError

import pytest  # type: ignore
from elgrad import Tensor, BroadcastError  # type: ignore


class TestConv2d:
    def _convolution(self, img: Tensor, filter: Tensor, stride: Union[int, tuple] = 1):
        result = img.conv2d(filter, stride=stride)
        sum = result.sum()
        sum.backward()
        return result, img.grad, filter.grad

    def test_one(self):
        img = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True, label="img").reshape(shape=(1, 1, 3, 3))
        filter = Tensor([[1, 2], [3, 4]], require_grad=True, label="filter").reshape(shape=(1, 1, 2, 2))
        result, img_grad, filter_grad = self._convolution(img, filter)

        conv_expected = Tensor([[[[37, 47], [67, 77]]]])
        img_grad_expected = Tensor([[[[1, 3, 2], [4, 10, 6], [3, 7, 4]]]])
        filter_grad_expected = Tensor([[[[12, 16], [24, 28]]]])
        assert Tensor.array_equal(result, conv_expected) and Tensor.array_equal(img_grad, img_grad_expected) and Tensor.array_equal(filter_grad, filter_grad_expected)

    def test_two(self):
        img = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], require_grad=True).reshape(shape=(1, 1, 4, 3))
        filter = Tensor([[1, 2], [3, 4]], require_grad=True).reshape(shape=(1, 1, 2, 2))
        result, img_grad, filter_grad = self._convolution(img, filter)

        conv_expected = Tensor([[[[37, 47], [67, 77], [97, 107]]]])
        img_grad_expected = Tensor([[[[1, 3, 2], [4, 10, 6], [4, 10, 6], [3, 7, 4]]]])
        filter_grad_expected = Tensor([[[[27, 33], [45, 51]]]])
        assert Tensor.array_equal(result, conv_expected) and Tensor.array_equal(img_grad_expected, img_grad) and Tensor.array_equal(filter_grad, filter_grad_expected)

    def test_three(self):
        img = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], require_grad=True).reshape(shape=(1, 1, 3, 4))
        filter = Tensor([[1, 2], [3, 4]], require_grad=True).reshape(shape=(1, 1, 2, 2))
        result, img_grad, filter_grad = self._convolution(img, filter)

        conv_expected = Tensor([[[[44, 54, 64], [84, 94, 104]]]])
        img_grad_expected = Tensor([[[[1, 3, 3, 2], [4, 10, 10, 6], [3, 7, 7, 4]]]])
        filter_grad_expected = Tensor([[[[24, 30], [48, 54]]]])

        assert Tensor.array_equal(result, conv_expected) and Tensor.array_equal(img_grad_expected, img_grad) and Tensor.array_equal(filter_grad, filter_grad_expected)

    def test_four(self):
        img = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], require_grad=True).reshape(shape=(1, 1, 4, 4))
        filter = Tensor([[1, 2], [3, 4]], require_grad=True).reshape(shape=(1, 1, 2, 2))
        result, img_grad, filter_grad = self._convolution(img, filter)

        conv_expected = Tensor([[[[44, 54, 64], [84, 94, 104], [124, 134, 144]]]])
        img_grad_expected = Tensor([[[[1, 3, 3, 2], [4, 10, 10, 6], [4, 10, 10, 6], [3, 7, 7, 4]]]])
        filter_grad_expected = Tensor([[[[54, 63], [90, 99]]]])

        assert Tensor.array_equal(result, conv_expected) and Tensor.array_equal(img_grad_expected, img_grad) and Tensor.array_equal(filter_grad, filter_grad_expected)

    def test_five(self):
        img = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True).reshape(shape=(1, 2, 3, 3))
        filter = Tensor([[1, 2], [3, 4], [1, 2], [3, 4]], require_grad=True).reshape(shape=(1, 2, 2, 2))
        result, img_grad, filter_grad = self._convolution(img, filter, stride=1)
        conv_expected = Tensor([[[[74, 94], [134, 154]]]])
        img_grad_expected = Tensor([[[[1, 3, 2], [4, 10, 6], [3, 7, 4]], [[1, 3, 2], [4, 10, 6], [3, 7, 4]]]])
        filter_grad_expected = Tensor([[[[12, 16], [24, 28]], [[12, 16], [24, 28]]]])

        # print(result, img_grad, filter_grad)
        assert Tensor.array_equal(result, conv_expected) and Tensor.array_equal(img_grad_expected, img_grad) and Tensor.array_equal(filter_grad, filter_grad_expected)


class TestMaxPooling:
    def _max_pooling(self, input: Tensor, filter_shape: tuple, stride: int):
        pooling_result = input.max_pooling(filter_shape, stride)
        sum_result = pooling_result.sum()
        sum_result.backward()
        return pooling_result, input.grad

    def test_one(self):
        img = Tensor([[4, 2, 3], [9, 5, 6], [1, 1, 1]], require_grad=True, label="img").reshape(shape=(1, 1, 3, 3))
        pooling_result, grad = self._max_pooling(img, (1, 1, 2, 2), 1)
        pooling_result_expected = Tensor([[[[9, 6], [9, 6]]]])
        grad_expected = Tensor([[[[0, 0, 0], [2, 0, 2], [0, 0, 0]]]])

        assert Tensor.array_equal(pooling_result, pooling_result_expected) and Tensor.array_equal(grad, grad_expected)

    def test_two(self):
        img = Tensor([[4, 2], [9, 5]], require_grad=True, label="img").reshape(shape=(1, 1, 2, 2))
        pooling_result, grad = self._max_pooling(img, (1, 1, 2, 2), 1)
        pooling_result_expected = Tensor([[[[9]]]])
        grad_expected = Tensor([[[[0, 0], [1, 0]]]])

        assert Tensor.array_equal(pooling_result, pooling_result_expected) and Tensor.array_equal(grad, grad_expected)

    def test_three(self):
        img = Tensor([[14, 2, 33, 8], [9, 55, 98, 6], [4, 2, 3, 8], [9, 55, 8, 6]], require_grad=True, label="img").reshape(shape=(1, 1, 4, 4))
        pooling_result, grad = self._max_pooling(img, (1, 1, 2, 2), 1)
        pooling_result_expected = Tensor([[[[55, 98, 98], [55, 98, 98], [55, 55, 8]]]])
        grad_expected = Tensor([[[[0, 0, 0, 0], [0, 2, 4, 0], [0, 0, 0, 1], [0, 2, 0, 0]]]])

        assert Tensor.array_equal(pooling_result, pooling_result_expected) and Tensor.array_equal(grad, grad_expected)

    def test_four(self):
        img = Tensor([[4, 2], [9, 5], [4, 2], [9, 5]], require_grad=True, label="img").reshape(shape=(1, 2, 2, 2))
        pooling_result, grad = self._max_pooling(img, (1, 2, 2, 2), 1)
        print(pooling_result, grad)
        pooling_result_expected = Tensor([[[[9]]]])
        grad_expected = Tensor([[[[0, 0], [1, 0]], [[0, 0], [0, 0]]]])

        assert Tensor.array_equal(pooling_result, pooling_result_expected) and Tensor.array_equal(grad, grad_expected)
