__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from functools import partial, reduce

import numpy as np
import torch

from arraycontext.fake_numpy import (
        BaseFakeNumpyNamespace, BaseFakeNumpyLinalgNamespace,
        )
from arraycontext.container import is_array_container
from arraycontext.container.traversal import (
        rec_map_array_container,
        rec_multimap_array_container,
        multimap_reduce_array_container,
        rec_map_reduce_array_container,
        rec_multimap_reduce_array_container,
        )


class TorchFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    # Everything is implemented in the base class for now.
    pass


class TorchFakeNumpyNamespace(BaseFakeNumpyNamespace):
    """
    A :mod:`numpy` mimic for :class:`~arraycontext.TorchArrayContext`.
    """
    def _get_fake_numpy_linalg_namespace(self):
        return TorchFakeNumpyLinalgNamespace(self._array_context)

    def __getattr__(self, name):
        return partial(rec_multimap_array_container, getattr(torch, name))

    # NOTE: the order of these follows the order in numpy docs
    # NOTE: when adding a function here, also add it to `array_context.rst` docs!

    # {{{ array creation routines

    def ones_like(self, ary):
        return self.full_like(ary, 1)

    def full_like(self, ary, fill_value):
        def _full_like(subary):
            return torch.full_like(subary, fill_value)

        return self._array_context._rec_map_container(
            _full_like, ary, default_scalar=fill_value)

    # }}}

    # {{{ array manipulation routines

    def reshape(self, a, newshape, order="C"):
        """
        .. warning::

            Since :func:`torch.reshape` does not support orders `A`` and
            ``K``, in such cases we fallback to using ``order = C``.
        """
        if order in "AK":
            from warnings import warn
            warn(f"reshape with order='{order}' nor supported by Torch,"
                 " using order=C.")
            
        return rec_map_array_container(
            lambda ary: torch.reshape(ary, newshape), a
        )

    def ravel(self, a, order="C"):
        """
        .. warning::

            Since :func:`torch.reshape` does not support orders `A`` and
            ``K``, in such cases we fallback to using ``order = C``.
        """
        if order in "AK":
            from warnings import warn
            warn(f"reshape with order='{order}' nor supported by Torch,"
                 " using order=C.")

        return rec_map_array_container(
            lambda ary: torch.ravel(ary), a
        )

    def transpose(self, a, dim0=0, dim1=1):
        return rec_multimap_array_container(torch.transpose, a, dim0, dim1)

    def broadcast_to(self, array, shape):
        return rec_map_array_container(partial(torch.broadcast_to, shape=shape), array)

    def concatenate(self, arrays, axis=0):
        return rec_multimap_array_container(torch.cat, arrays, axis)

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
            lambda *args: torch.stack(tensors=args, dim=axis), *arrays)

    # {{{ linear algebra

    # }}}

    # {{{ logic functions

    def all(self, a):
        return rec_map_reduce_array_container(
            partial(reduce, torch.logical_and), torch.all, a)

    def any(self, a):
        return rec_map_reduce_array_container(
            partial(reduce, torch.logical_or), torch.any, a)

    def array_equal(self, a, b):
        actx = self._array_context

        true = actx.from_numpy(np.int8(True))
        false = actx.from_numpy(np.int8(False))

        def rec_equal(x, y):
            if type(x) != type(y):
                return false

            try:
                iterable = zip(serialize_container(x), serialize_container(y))
            except NotAnArrayContainerError:
                if x.shape != y.shape:
                    return false
                else:
                    return torch.all(torch.equal(x, y))
            else:
                return reduce(
                    torch.logical_and,
                    [rec_equal(ix, iy) for (_, ix), (_, iy) in iterable],
                    true)

        return rec_equal(a, b)

    def equal(self, a, b):
        # ECG: Really?        
        return a == b
                     
    # }}}
    
    # {{{ mathematical functions

    def sum(self, a, axis=0, dtype=None):        
        return rec_map_reduce_array_container(
            sum,
            partial(torch.sum, axis=axis, dtype=dtype),
            a)

    def amin(self, a, axis=0):
        return rec_map_reduce_array_container(
            partial(reduce, torch.minimum), partial(torch.amin, axis=axis), a)

    min = amin

    def amax(self, a, axis=0):
        return rec_map_reduce_array_container(
            partial(reduce, torch.maximum), partial(torch.amax, axis=axis), a)

    max = amax
    
    # }}}

    # {{{ sorting, searching, and counting

    def where(self, criterion, then, else_):
        def where_inner(inner_crit, inner_then, inner_else):
            import torch
            if isinstance(inner_crit, torch.BoolTensor):
                return torch.where(inner_crit, inner_then, inner_else)
            else:
                return torch.where(inner_crit != 0, inner_then, inner_else)
            
        return rec_multimap_array_container(where_inner, criterion, then, else_)
    
    # }}}
