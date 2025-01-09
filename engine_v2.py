from typing import Union, List, Tuple

import numpy as np #type: ignore

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = []
        self._get_shape(data)
        self._current_index = 0

    def __getitem__(self, indices: Union[List, int, Tuple, slice]):
       return self.data[indices]

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index > self.shape[0] - 1:
            raise StopIteration

        tensor = Tensor(self.data[self.current_index])
        self.current_index += 1
        return tensor

    def __add__(self, other):
        assert self.shape == other.shape, "Different shape"
        result = self.data + other.data

        return Tensor(result)

    def __mul__(self, other):
        return Tensor(np.matmul(self.data, other.data))

    def _get_shape(self, x):
        if not hasattr(x, "__len__"):
            return
        else:
            self.shape.append(len(x))
            return self._get_shape(x[0])

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, shape={self.shape})"


if __name__ == "__main__":
    a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    c = Tensor([[1, 2], [3, 4], [5, 6]])
    
    print(a * c)

