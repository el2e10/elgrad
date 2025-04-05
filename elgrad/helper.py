from typing import Union, Tuple

def check_conv2d_stride_shape(inputr, filter, stride: Union[int, Tuple[int]]) -> Tuple[bool, str]:
    if(not isinstance(stride, int | tuple)):
        return False, "Stride should be an int or tuple"
    if(isinstance(stride, int) and (stride > min(filter.shape))):
        return False, "Invalid stride length"
    if(isinstance(stride, tuple) and (len(stride) != filter.ndim)):
        return False, "Invalid stride dimension"
    return True, ""
