from math import e
from typing import Union

from elgrad.tensor import DotProductError

import pytest  # type: ignore
from elgrad import Tensor, BroadcastError  # type: ignore


class TestConv2d:

    def _convolution(self, img: Tensor, filter: Tensor, stride=1):
        result = img.conv2d(filter, stride=stride)
        sum = result.sum()
        sum.backward()
        return result, img.grad, filter.grad

    def test_one(self):
        img = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True, label="img").reshape(shape=(1, 1, 3, 3))
        filter = Tensor([[1, 2], [3, 4]], require_grad=True, label="filter").reshape(shape=(1, 1, 2, 2))
        result, img_grad, filter_grad = self._convolution(img, filter) 
        print(result, img_grad, filter_grad)

        conv_expected = Tensor([[37, 47], [67, 77]])
        img_grad_expected = Tensor([[[[1, 3, 2], [4, 10, 6], [3, 7, 4]]]])
        filter_grad_expected = Tensor([[[[12, 16], [24, 28]]]])
        assert Tensor.array_equal(result, conv_expected) and Tensor.array_equal(img_grad, img_grad_expected) and Tensor.array_equal(filter_grad, filter_grad_expected)

    '''
    def test_two(self):
        img = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], require_grad=True)
        filter = Tensor([[1, 2], [3, 4]], require_grad=True)
        result, img_grad, filter_grad = self._convolution(img, filter) 

        conv_expected = Tensor([[37, 47], [67, 77], [97, 107]])
        img_grad_expected = Tensor([[1, 3, 2], [4, 10, 6], [4, 10, 6], [3, 7, 4]])
        filter_grad_expected = Tensor([[27, 33], [45, 51]])
        assert Tensor.array_equal(result, conv_expected) and Tensor.array_equal(img_grad_expected, img_grad) and Tensor.array_equal(filter_grad, filter_grad_expected)


    def test_three(self):
        img = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], require_grad=True)
        filter = Tensor([[1, 2], [3, 4]], require_grad=True)
        result, img_grad, filter_grad = self._convolution(img, filter) 

        conv_expected = Tensor([[44, 54, 64], [84, 94, 104]])
        img_grad_expected = Tensor([[1, 3, 3, 2], [4, 10, 10, 6], [3, 7, 7, 4]])
        filter_grad_expected = Tensor([[24, 30], [48, 54]])

        assert Tensor.array_equal(result, conv_expected) and Tensor.array_equal(img_grad_expected, img_grad) and Tensor.array_equal(filter_grad, filter_grad_expected)


    def test_four(self):
        img = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], require_grad=True)
        filter = Tensor([[1, 2], [3, 4]], require_grad=True)
        result, img_grad, filter_grad = self._convolution(img, filter) 
        
        conv_expected = Tensor([[44, 54, 64], [84, 94, 104], [124, 134, 144]])
        img_grad_expected = Tensor([[1, 3, 3, 2], [4, 10, 10, 6], [4, 10, 10, 6], [3, 7, 7, 4]])
        filter_grad_expected = Tensor([[54, 63], [90, 99]])

        assert Tensor.array_equal(result, conv_expected) and Tensor.array_equal(img_grad_expected, img_grad) and Tensor.array_equal(filter_grad, filter_grad_expected)
    
    def test_five(self):
        img = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], require_grad=True)
        filter = Tensor([[1, 2], [3, 4]], require_grad=True)
        result, img_grad, filter_grad = self._convolution(img, filter, stride=2) 

        conv_expected = Tensor([[44, 64], [124, 144]])
        img_grad_expected = Tensor([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])
        filter_grad_expected = Tensor([[24, 28], [40, 44]])

        print(result)

        assert Tensor.array_equal(result, conv_expected) and Tensor.array_equal(img_grad_expected, img_grad) and Tensor.array_equal(filter_grad, filter_grad_expected)

'''
