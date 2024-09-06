__copyright__ = """
Copyright (C) 2024 University of Illinois Board of Trustees
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

import cupy as cp

from arraycontext.container import NotAnArrayContainerError, serialize_container
from arraycontext.container.traversal import (
    rec_map_array_container,
    rec_map_reduce_array_container,
    rec_multimap_array_container,
    rec_multimap_reduce_array_container,
)
from arraycontext.context import Array, ArrayOrContainer
from arraycontext.fake_numpy import (
    BaseFakeNumpyLinalgNamespace,
    BaseFakeNumpyNamespace,
)


class CupyFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    # Everything is implemented in the base class for now.
    pass


_NUMPY_UFUNCS = frozenset({"concatenate", "reshape", "transpose",
                 "where",
                 *BaseFakeNumpyNamespace._numpy_math_functions
                 })


class CupyFakeNumpyNamespace(BaseFakeNumpyNamespace):
    """
    A :mod:`cupy` mimic for :class:`CupyArrayContext`.
    """
    def _get_fake_numpy_linalg_namespace(self):
        return CupyFakeNumpyLinalgNamespace(self._array_context)

    def zeros(self, shape, dtype):
        return cp.zeros(shape, dtype)

    def __getattr__(self, name):

        if name in _NUMPY_UFUNCS:
            from functools import partial
            return partial(rec_multimap_array_container,
                           getattr(cp, name))

        raise AttributeError(name)

    def sum(self, a, axis=None, dtype=None):
        return rec_map_reduce_array_container(sum, partial(cp.sum,
                                                           axis=axis,
                                                           dtype=dtype),
                                              a)

    def min(self, a, axis=None):
        return rec_map_reduce_array_container(
                partial(reduce, cp.minimum), partial(cp.amin, axis=axis), a)

    def max(self, a, axis=None):
        return rec_map_reduce_array_container(
                partial(reduce, cp.maximum), partial(cp.amax, axis=axis), a)

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
                lambda *args: cp.stack(args, axis=axis),
                *arrays)

    def broadcast_to(self, array, shape):
        return rec_map_array_container(partial(cp.broadcast_to, shape=shape), array)

    # {{{ relational operators

    def equal(self, x, y):
        return rec_multimap_array_container(cp.equal, x, y)

    def not_equal(self, x, y):
        return rec_multimap_array_container(cp.not_equal, x, y)

    def greater(self, x, y):
        return rec_multimap_array_container(cp.greater, x, y)

    def greater_equal(self, x, y):
        return rec_multimap_array_container(cp.greater_equal, x, y)

    def less(self, x, y):
        return rec_multimap_array_container(cp.less, x, y)

    def less_equal(self, x, y):
        return rec_multimap_array_container(cp.less_equal, x, y)

    # }}}

    def ravel(self, a, order="C"):
        return rec_map_array_container(partial(cp.ravel, order=order), a)

    def vdot(self, x, y):
        return rec_multimap_reduce_array_container(sum, cp.vdot, x, y)

    def any(self, a):
        return rec_map_reduce_array_container(partial(reduce, cp.logical_or),
                                              lambda subary: cp.any(subary), a)

    def all(self, a):
        return rec_map_reduce_array_container(partial(reduce, cp.logical_and),
                                              lambda subary: cp.all(subary), a)

    def array_equal(self, a: ArrayOrContainer, b: ArrayOrContainer) -> Array:
        false_ary = cp.array(False)
        true_ary = cp.array(True)
        if type(a) is not type(b):
            return false_ary

        try:
            serialized_x = serialize_container(a)
            serialized_y = serialize_container(b)
        except NotAnArrayContainerError:
            assert isinstance(a, cp.ndarray)
            assert isinstance(b, cp.ndarray)
            return cp.array(cp.array_equal(a, b))
        else:
            if len(serialized_x) != len(serialized_y):
                return false_ary
            return reduce(
                    cp.logical_and,
                    [(true_ary if kx_i == ky_i else false_ary)
                        and self.array_equal(x_i, y_i)
                        for (kx_i, x_i), (ky_i, y_i)
                        in zip(serialized_x, serialized_y)],
                    true_ary)

    def arange(self, *args, **kwargs):
        return cp.arange(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        return cp.linspace(*args, **kwargs)

    def zeros_like(self, ary):
        if isinstance(ary, (int, float, complex)):
            # Cupy does not support zeros_like with scalar arguments
            ary=cp.array(ary)
        return rec_map_array_container(cp.zeros_like, ary)

    def ones_like(self, ary):
        if isinstance(ary, (int, float, complex)):
            # Cupy does not support ones_like with scalar arguments
            ary=cp.array(ary)
        return rec_map_array_container(cp.ones_like, ary)

    def reshape(self, a, newshape, order="C"):
        return rec_map_array_container(
                lambda ary: ary.reshape(newshape, order=order),
                a)


# vim: fdm=marker
