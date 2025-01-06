from typing import Union, List, Tuple

class Tensor:

    def __init__(self, data):
        self.data = data
        self.shape = [] 
        self._get_shape(data)

    def __getitem__(self, indices: Union[List, int, Tuple]):
        # To handle indexing like t[0]
        if isinstance(indices, int):
            return self.data[indices]
        if not hasattr(indices, "__len__") and not isinstance(indices, int):
            raise TypeError("Invalid type for index")

        if len(indices) > len(self.shape):
            raise IndexError("Invalid dimension for the Tensor")
        if not all([isinstance(x, int) for x in indices]):
            raise TypeError("Invalid type for index")

        # To handle indexing like t[1, 2, 3]
        result = self.data
        for index in indices:
            result = result[index]
        return result

    def __add__(self, other):
        assert self.shape == other.shape, "Different shape"
        return other

    def _get_shape(self, x):
        if(not hasattr(x, "__len__")):
            return 
        else:
            self.shape.append(len(x))
            return self._get_shape(x[0])

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, shape={self.shape})"

if __name__ == "__main__":
    a = [[1, 2, 3],[2,3,8]]
    b = Tensor([[4, 5, 7],[2, 3, 5]])

        


