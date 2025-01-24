from elgrad import Tensor #type: ignore


class TestMatAdd():
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
        a_grad_expected, b_grad_expected = Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected) 

    def test_two(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([[10, 20, 30]], require_grad=True)

        a_grad, b_grad = self.matadd_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), Tensor([[3, 3, 3]])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected) 

    def test_three(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([10, 20, 30], require_grad=True)

        a_grad, b_grad = self.matadd_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), Tensor([3, 3, 3])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected) 

    def test_four(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], require_grad=True)
        b = Tensor([[10], [20]], require_grad=True)

        a_grad, b_grad = self.matadd_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([[1, 1, 1], [1, 1, 1]]), Tensor([[3], [3]])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected) 




class TestMatMul():
    def matmul_grad(self, x: Tensor, y: Tensor):
        c:Tensor = x @ y
        d = c.sum()
        d.backward()
        return x.grad, y.grad

    def test_one(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([1, 2, 3], require_grad=True)

        a_grad, b_grad = self.matmul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), Tensor([12, 15, 18])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected) 

    def test_two(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], require_grad=True)
        b = Tensor([44, 55, 66], require_grad=True)

        a_grad, b_grad = self.matmul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([[44, 55, 66], [44, 55, 66]]), Tensor([5, 7, 9])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected) 

    def test_three(self):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True)
        b = Tensor([[1, 2, 3], [11, 22, 33], [44, 55, 66]], require_grad=True)

        a_grad, b_grad = self.matmul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([[6, 66, 165], [6, 66, 165],[6, 66, 165]]), Tensor([[12, 12, 12], [15, 15, 15], [18, 18, 18]])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected) 

    def test_four(self):
        a = Tensor([[1, 2], [3, 4], [5, 6]], require_grad=True)
        b = Tensor([[7, 8, 9, 10], [11, 12, 13, 14]], require_grad=True)

        a_grad, b_grad = self.matmul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([[34, 50], [34, 50],[34, 50]]), Tensor([[9, 9, 9, 9], [12, 12, 12, 12]])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected) 

    def test_five(self):
        a = Tensor([[1], [2], [3], [4], [5]], require_grad=True)
        b = Tensor([[6, 7, 8]], require_grad=True)

        a_grad, b_grad = self.matmul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([[21],[21],[21],[21],[21]]), Tensor([[15, 15, 15]])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected) 

    def test_six(self):
        a = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]], require_grad=True)
        b = Tensor([9, 8], require_grad=True)

        a_grad, b_grad = self.matmul_grad(a, b)
        print(a_grad, b_grad)
        a_grad_expected, b_grad_expected = Tensor([[9, 8],[9, 8],[9, 8],[9, 8]]), Tensor([16, 20])
        assert Tensor.array_equal(a_grad, a_grad_expected) and Tensor.array_equal(b_grad, b_grad_expected) 







