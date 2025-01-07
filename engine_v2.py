from typing import Union, List, Tuple

class Tensor:

    def __init__(self, data):
        self.data = data
        self.shape = []
        self._get_shape(data)
        self._current_index = 0

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
        return Tensor(result)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if(self.current_index > self.shape[0]-1):
            raise StopIteration

        tensor = Tensor(self.data[self.current_index])
        self.current_index += 1
        return tensor

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
    a = Tensor([[[1, 2, 3],[2, 3, 8]], [[4, 8, 9], [4, 7, 9]]])
    b = Tensor([[4, 5, 7],[2, 3, 5]])

    for i in a:
        for j in i:
            print(j)
    
    # result = []
    # def sum(a, b):
    #     print(a)
    #     if(len(a) == len(b) == 1):
    #         return [x + y for x, y in zip(a,b)]
    #     else:
    #         result.append(sum([*a], [*b]))
    #
    # sum(a, b)
    # print(result)


