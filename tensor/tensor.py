import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = 0
        self.children = []
        
        
    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()


def tensor(data: np.ndarray):
    return Tensor(data)

def ones(shape: tuple):
    return Tensor(np.ones(shape))

def zeros(shape: tuple):
    return Tensor(np.zeros(shape))
