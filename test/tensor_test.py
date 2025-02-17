from typing import Union

import pytest  # type: ignore
from elgrad import Tensor, BroadcastError  # type: ignore

class TestReLU:
    def relu_grad(self, x: Tensor):
        c = x.relu()
        d = c.sum()
        d.backward()
        return x.grad

        
    def test_one(self):
        a = Tensor([1, 2, 3], require_grad=True)
        a_grad = self.relu_grad(a)
        a_grad_expected = Tensor([1, 1, 1])
        print(a_grad)
        assert Tensor.array_equal(a_grad, a_grad_expected)

    def test_two(self):
        a = Tensor([[1, 2, 3], [0, -2, 3]], require_grad=True)
        a_grad = self.relu_grad(a)
        a_grad_expected = Tensor([[1, 1, 1], [0, 0, 1]])
        assert Tensor.array_equal(a_grad, a_grad_expected)

    def test_three(self):
        a = Tensor([[1, 0, 3], [1, 2, 3]], require_grad=True)
        a_grad = self.relu_grad(a)
        a_grad_expected = Tensor([[1, 0, 1], [1, 1, 1]])
        assert Tensor.array_equal(a_grad, a_grad_expected)

class TestSoftmax:
    def softmax_grad(self, x: Tensor, dim: int):
        b = x.softmax(dim)
        return b

    def test_one(self):
        a = Tensor([1, 2, 3], require_grad=True)
        a_grad = self.softmax_grad(a, 0)
        print(a_grad)
        a_grad_expected = Tensor([.0900, .2447, .6652])
        assert Tensor.array_equal(a_grad, a_grad_expected)

    def test_two(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], require_grad=True)
        a_grad = self.softmax_grad(a, 0)
        print(a_grad)
        a_grad_expected = Tensor([[0.0474, 0.0474, 0.0474], [0.9526, 0.9526, 0.9526]])
        assert Tensor.array_equal(a_grad, a_grad_expected)

    def test_three(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], require_grad=True)
        a_grad = self.softmax_grad(a, 1)
        print(a_grad)
        a_grad_expected = Tensor([[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]])
        assert Tensor.array_equal(a_grad, a_grad_expected)

    def test_four(self):
        a = Tensor([[[1.,2.,3.], [4, 5, 6]], [[4.,5.,6.], [1, 2, 3]]], require_grad=True)
        a_grad = self.softmax_grad(a, 1)
        print(a_grad)
        a_grad_expected = Tensor([[[0.0474, 0.0474, 0.0474], [0.9526, 0.9526, 0.9526]], [[0.9526, 0.9526, 0.9526], [0.0474, 0.0474, 0.0474]]])
        assert Tensor.array_equal(a_grad, a_grad_expected)

    def test_five(self):
        a = Tensor([[[1.,2.,3.], [4, 5, 6]], [[1.,2.,3.], [4, 5, 6]]], require_grad=True)
        a_grad = self.softmax_grad(a, 1)
        print(a_grad)
        a_grad_expected = Tensor([[[0.0474, 0.0474, 0.0474], [0.9526, 0.9526, 0.9526]], [[0.0474, 0.0474, 0.0474], [0.9526, 0.9526, 0.9526]]])
        assert Tensor.array_equal(a_grad, a_grad_expected)

    def test_six(self):
        a = Tensor([[[1.,2.,3.], [4, 5, 6]], [[1.,2.,3.], [4, 5, 6]]], require_grad=True)
        a_grad = self.softmax_grad(a, 2)
        print(a_grad)
        a_grad_expected = Tensor([[[0.09, .2447, .6652], [0.0900, .2447, .6652]], [[0.0900, .2447, .6652], [.0900, .2447, .6652]]])
        assert Tensor.array_equal(a_grad, a_grad_expected)



class TestMatLog:
    def mat_log_grad(self, x: Tensor):
        c = x.log()
        d = c.sum()
        d.backward()
        return x.grad

    def mat_log_grad_2(self, x: Tensor, y: Tensor):
        b = x.log()
        c = b + y
        d = c.sum()
        d.backward()
        return x.grad, y.grad 

    def mat_log_grad_3(self, x: Tensor, y: Tensor):
        b = x.log()
        c = b * y
        print(c)
        d = c.sum()
        print(d)
        d.backward()
        return x.grad, y.grad 
    
    def test_one(self):
        a = Tensor([1, 2, 3], require_grad=True)
        a_grad = self.mat_log_grad(a)
        print(a_grad)
        a_grad_expected = Tensor([1., .5, .3333])
        assert Tensor.array_equal(a_grad, a_grad_expected)

    def test_two(self):
        a = Tensor([[1, 2, 3], [2, 3, 4]], require_grad=True)
        a_grad = self.mat_log_grad(a)
        print(a_grad)
        a_grad_expected = Tensor([[1., .5, .3333], [.5, .3333, .25]])
        assert Tensor.array_equal(a_grad, a_grad_expected)

    def test_three(self):
        a = Tensor([[1, 2, 3], [2, 3, 4]], require_grad=True)
        b = Tensor([[1, 2, 3]], require_grad=True)
        a_grad, b_grad = self.mat_log_grad_2(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([[1., .5, .3333], [.5, .3333, .25]]), Tensor([[2, 2, 2]])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected)

    def test_four(self):
        a = Tensor([[1, 2, 3], [2, 3, 4]], require_grad=True)
        b = Tensor([1, 2, 3], require_grad=True)
        a_grad, b_grad = self.mat_log_grad_2(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([[1., .5, .3333], [.5, .3333, .25]]), Tensor([2, 2, 2])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected)

    def test_five(self):
        a = Tensor([[1, 2, 3], [2, 3, 4]], require_grad=True)
        b = Tensor([1, 2, 3], require_grad=True)
        a_grad, b_grad = self.mat_log_grad_3(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([[1., 1., 1.], [.5, .6667, .75]]), Tensor([.6931, 1.7918, 2.4849])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected)


class TestMatPow:
    def mat_pow_grad(self, x: Tensor, exponenet: Tensor):
        c = x**exponenet
        d = c.sum()
        d.backward()
        return x.grad, exponenet.grad

    def test_one(self):
        a = Tensor([1, 2, 3], require_grad=True)
        b = Tensor([2], require_grad=True)

        a_grad, b_grad = self.mat_pow_grad(a, b)
        a_grad_expected = Tensor([2, 4, 6])
        b_grad_expected = Tensor([12.6601])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected)

    def test_two(self):
        a = Tensor([1, 2, 3], require_grad=True)
        b = Tensor([1, 2, 3], require_grad=True)

        a_grad, b_grad = self.mat_pow_grad(a, b)#type: ignore
        a_grad_expected, b_grad_expected = Tensor([1, 4, 27]), Tensor([0, 2.7726, 29.6625])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected)

    def test_three(self):
        a = Tensor([[1, 2, 3],[1, 2, 3]], require_grad=True)
        b = Tensor([1, 2, 3], require_grad=True)

        a_grad, b_grad = self.mat_pow_grad(a, b)#type: ignore
        a_grad_expected, b_grad_expected = Tensor([[1, 4, 27], [1, 4, 27]]), Tensor([0, 5.5452, 59.3251])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected)

    def test_four(self):
        a = Tensor([[1, 2, 3],[1, 2, 3]], require_grad=True)
        b = Tensor([[1, 2, 3], [2, 2, 4]], require_grad=True)

        a_grad, b_grad = self.mat_pow_grad(a, b)#type: ignore
        a_grad_expected, b_grad_expected = Tensor([[1, 4, 27], [2, 4, 108]]), Tensor([[0, 2.7726, 29.6625],[0, 2.7726, 88.9876]])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected)

    def test_five(self):
        a = Tensor([[1, 2, 3],[1, 2, 3], [2, 2, 4]], require_grad=True)
        b = Tensor([[1, 2, 3], [2, 2, 4]], require_grad=True)

        with pytest.raises(BroadcastError):
            a_grad, b_grad = self.mat_pow_grad(a, b)#type: ignore


class TestMatElemDiv:
    def mat_elem_div_grad(self, x: Tensor, y: Tensor):
        c = x / y
        d = c.sum()
        d.backward()
        return x.grad, y.grad

    def check_div(self, x: Union[Tensor, float, int], y: Union[Tensor, float, int]):
        c = x / y
        return c

    def test_one(self):
        a = Tensor([1, 2, 3], require_grad=True)
        b = Tensor([1, 2, 3], require_grad=True)

        a_grad, b_grad = self.mat_elem_div_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([1.0000, 0.5000, 0.3333]),
            Tensor([-1.0000, -0.5000, -0.3333]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_two(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], require_grad=True)

        a_grad, b_grad = self.mat_elem_div_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor(
                [
                    [0.1000, 0.0500, 0.0333],
                    [0.0250, 0.0200, 0.0167],
                    [0.0143, 0.0125, 0.0111],
                ]
            ),
            Tensor(
                [
                    [-0.0100, -0.0050, -0.0033],
                    [-0.0025, -0.0020, -0.0017],
                    [-0.0014, -0.0012, -0.0011],
                ]
            ),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_three(self):
        a = Tensor([[1, 2], [3, 4]], require_grad=True)
        b = Tensor([[5, 6, 7], [8, 9, 10]], require_grad=True)

        with pytest.raises(BroadcastError):
            _, _ = self.mat_elem_div_grad(a, b)

    def test_four(self):
        a = Tensor([[10, 11, 12]], require_grad=True)
        b = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)

        a_grad, b_grad = self.mat_elem_div_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[1.3929, 0.8250, 0.6111]]),
            Tensor(
                [
                    [-10.0000, -2.7500, -1.3333],
                    [-0.6250, -0.4400, -0.3333],
                    [-0.2041, -0.1719, -0.1481],
                ]
            ),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_five(self):
        a = Tensor([1], require_grad=True)
        b = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)

        a_grad, b_grad = self.mat_elem_div_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([2.829]),
            Tensor(
                [
                    [-1, -0.25, -0.1111],
                    [-0.0625, -0.0400, -0.0278],
                    [-0.0204, -0.0156, -0.0123],
                ]
            ),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_six(self):
        a = 1
        b = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)

        result = self.check_div(a, b)
        result_expected = Tensor([[1.        , 0.5       , 0.33333333], [0.25      , 0.2       , 0.16666667], [0.14285714, 0.125     , 0.11111111]])
        print(result, result_expected)
        assert Tensor.array_equal(result, result_expected) 

class TestMatElemMul:
    def mat_elem_mul_grad(self, x: Tensor, y: Tensor):
        c = x * y
        print(f"Mat mul result is {c}")
        d = c.sum()
        d.backward()
        return x.grad, y.grad

    def test_one(self):
        a = Tensor([1, 2, 3], require_grad=True)
        b = Tensor([4, 5, 6], require_grad=True)

        a_grad, b_grad = self.mat_elem_mul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([4, 5, 6]), Tensor([1, 2, 3])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_two(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], require_grad=True)

        a_grad, b_grad = self.mat_elem_mul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]]),
            Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_three(self):
        a = Tensor([[1, 2], [3, 4]], require_grad=True)
        b = Tensor([[5, 6, 7], [8, 9, 10]], require_grad=True)

        with pytest.raises(BroadcastError):
            _, _ = self.mat_elem_mul_grad(a, b)

    def test_four(self):
        a = Tensor([1, 2, 3], require_grad=True)
        b = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)

        a_grad, b_grad = self.mat_elem_mul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([12, 15, 18]),
            Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_five(self):
        a = Tensor([1], require_grad=True)
        b = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)

        a_grad, b_grad = self.mat_elem_mul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([45]),
            Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )


class TestMatAdd:
    def matadd_grad(self, x: Tensor, y: Tensor):
        c = x + y
        print(f"Mat add result is {c}")
        d = c.sum()
        d.backward()
        return x.grad, y.grad

    def test_one(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([[11, 12, 13], [14, 15, 16], [17, 18, 19]], require_grad=True)

        a_grad, b_grad = self.matadd_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_two(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([[10, 20, 30]], require_grad=True)

        a_grad, b_grad = self.matadd_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            Tensor([[3, 3, 3]]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_three(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([10, 20, 30], require_grad=True)

        a_grad, b_grad = self.matadd_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            Tensor([3, 3, 3]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_four(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], require_grad=True)
        b = Tensor([[10], [20]], require_grad=True)

        a_grad, b_grad = self.matadd_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[1, 1, 1], [1, 1, 1]]),
            Tensor([[3], [3]]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )


class TestMatSub:
    def matadd_grad(self, x: Tensor, y: Tensor):
        c = x - y
        print(f"Mat sub result is {c}")
        d = c.sum()
        d.backward()
        return x.grad, y.grad

    def test_one(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([[11, 12, 13], [14, 15, 16], [17, 18, 19]], require_grad=True)

        a_grad, b_grad = self.matadd_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            Tensor([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_two(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([[10, 20, 30]], require_grad=True)

        a_grad, b_grad = self.matadd_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            Tensor([[-3, -3, -3]]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_three(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([10, 20, 30], require_grad=True)

        a_grad, b_grad = self.matadd_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            Tensor([-3, -3, -3]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_four(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], require_grad=True)
        b = Tensor([[10], [20]], require_grad=True)

        a_grad, b_grad = self.matadd_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[1, 1, 1], [1, 1, 1]]),
            Tensor([[-3], [-3]]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )


class TestMatMul:
    def matmul_grad(self, x: Tensor, y: Tensor):
        c: Tensor = x @ y
        d = c.sum()
        d.backward()
        return x.grad, y.grad

    def test_one(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([1, 2, 3], require_grad=True)

        a_grad, b_grad = self.matmul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
            Tensor([12, 15, 18]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_two(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], require_grad=True)
        b = Tensor([44, 55, 66], require_grad=True)

        a_grad, b_grad = self.matmul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[44, 55, 66], [44, 55, 66]]),
            Tensor([5, 7, 9]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_three(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([[1, 2, 3], [11, 22, 33], [44, 55, 66]], require_grad=True)

        a_grad, b_grad = self.matmul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[6, 66, 165], [6, 66, 165], [6, 66, 165]]),
            Tensor([[12, 12, 12], [15, 15, 15], [18, 18, 18]]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_four(self):
        a = Tensor([[1, 2], [3, 4], [5, 6]], require_grad=True)
        b = Tensor([[7, 8, 9, 10], [11, 12, 13, 14]], require_grad=True)

        a_grad, b_grad = self.matmul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[34, 50], [34, 50], [34, 50]]),
            Tensor([[9, 9, 9, 9], [12, 12, 12, 12]]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_five(self):
        a = Tensor([[1], [2], [3], [4], [5]], require_grad=True)
        b = Tensor([[6, 7, 8]], require_grad=True)

        a_grad, b_grad = self.matmul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[21], [21], [21], [21], [21]]),
            Tensor([[15, 15, 15]]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )

    def test_six(self):
        a = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]], require_grad=True)
        b = Tensor([9, 8], require_grad=True)

        a_grad, b_grad = self.matmul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = (
            Tensor([[9, 8], [9, 8], [9, 8], [9, 8]]),
            Tensor([16, 20]),
        )
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(
            b_grad, b_grad_expected
        )
