import numpy as np
from copy import deepcopy
from typing import Union
from collections.abc import Iterable
import numpy as np
import os


def indicator_func_to_matrix(size,indicator_func,dtype=bool):
    A = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            A[i,j] = indicator_func(i,j)
    return A.astype(dtype)

def make_path_if_not_exist(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def get_env_var(variable_name='ROOT_PATH'):
    env_var = os.getenv(variable_name)
    if not env_var:
        raise EnvironmentError(f"'{variable_name}' environment variable not set")
    return env_var

def duplicate_list(item,repeats):
    return [item] * repeats

def set_attributes(object, **kwargs):
    if not len(kwargs):
        return object
    for attr,val in kwargs.items(): #zip(attributes,values):
        setattr(object,attr,val)
    return object

def copy_attributes(from_object,to_object,attributes):
    for attr in attributes:
        setattr(
            to_object,
            attr,
            getattr(from_object,attr)
        )
    return to_object

def empty_call(func):
    return func()

def select(obj,**kwargs):
    return obj.select(**kwargs)


# setattrs = lambda target, attributes, values:  [setattr(target,attr,val) for attr,val in zip(attributes,values)]
def setattrs(target,attributes,values):
    for attr,value in zip(attributes,values):
        setattr(target,attr,value)
    return target
setattrs_kwargs = lambda target,**kwargs: setattrs(target,*list(zip(*kwargs.items())))

#def flatten_with_lengths_xarray(xarray):



# def __init__(self,ndarray):
#     super().__init__()
#     self.array = ndarray
#
# def split(self,indices,axis=None):
#     return np.split(self,indices,axis=axis)


batch_choice = lambda p:  np.stack([np.random.choice(p.shape[1],p=p[i]) for i in range(len(p))])

normalise = lambda array,axis=None: array/array.sum(axis=axis,keepdims=True)


if __name__ == '__main__':
    pass