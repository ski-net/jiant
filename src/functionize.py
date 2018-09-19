import torch
from torch import nn
from torch.nn import functional as F

class Scope(object):
    pass

import sys
from collections import OrderedDict

PY2 = sys.version_info[0] == 2
_internal_attrs = {'_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_modules'}


class Scope(object):
    def __init__(self):
        self._modules = OrderedDict()


def _make_functional(module, params_box, params_offset):
    ''' TODO(Alex): something like create a functional version of module
    1) Create a Scope object
    2) "Copy" the module by setting Scope's attributes to module attributes
        a) Recursively apply _make_functional to named children
    3) Create functional fmodule and return it.
        fmodule
    '''
    self = Scope()
    num_params = len(module._parameters)
    param_names = list(module._parameters.keys())
    forward = type(module).forward.__func__ if PY2 else type(module).forward
    for name, attr in module.__dict__.items():
        if name in _internal_attrs:
            continue
        setattr(self, name, attr)
    if hasattr(module, '_forward_funcs'):
        for name, ffunc in module._forward_funcs.items():
            setattr(self, name, ffunc)

    child_params_offset = params_offset + num_params
    for name, child in module.named_children():
        child_params_offset, fchild = _make_functional(child, params_box, child_params_offset)
        self._modules[name] = fchild
        setattr(self, name, fchild)

    def fmodule(*args, **kwargs):
        ''' Create a closure using param_names, params_offset, num_params, forward.
        I think because params_box is a list, it persists in memory and
        the parameters will be there when it is modified by fmodule
        when the parameters are popped and set.  '''
        for name, param in zip(param_names, params_box[0][params_offset:params_offset + num_params]):
            setattr(self, name, param)
        return forward(self, *args, **kwargs)

    return child_params_offset, fmodule


def make_functional(module):
    ''' Given an nn.Module, return a functional version, fmodule, of that module.
    The functional expects the input expected by module.forward, but also
    accepts an additional keyword `params`, which is used ...
    '''

    # params_box holds parameters
    # when constructing fmodule, params_box = [None]
    params_box = [None]
    _, fmodule_internal = _make_functional(module, params_box, 0)

    def fmodule(*args, **kwargs):
        ''' Functional that actually gets returned
        Will get called as forward(x, params=theta) '''
        params_box[0] = kwargs.pop('params')
        return fmodule_internal(*args, **kwargs)

    return fmodule


