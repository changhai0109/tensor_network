from . import Tensor

def place_holder(dimensions_symbol, label, require_grads=False):
    return Tensor(dimensions_symbol, label, require_grads)
