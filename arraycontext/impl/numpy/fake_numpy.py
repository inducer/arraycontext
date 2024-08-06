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


class NumpyFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    # Everything is implemented in the base class for now.
    pass


_NUMPY_UFUNCS = {"abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
                 "sinh", "cosh", "tanh", "exp", "log", "log10", "isnan",
                 "sqrt", "concatenate", "reshape", "transpose",
                 "ones_like", "maximum", "minimum", "where", "conj", "arctan2",
                 }


class NumpyFakeNumpyNamespace(BaseFakeNumpyNamespace):
    """
    A :mod:`numpy` mimic for :class:`NumpyArrayContext`.
    """
    def _get_fake_numpy_linalg_namespace(self):
        return NumpyFakeNumpyLinalgNamespace(self._array_context)

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype)

    def __getattr__(self, name):

        if name in _NUMPY_UFUNCS:
            from functools import partial
            return partial(rec_multimap_array_container,
                           getattr(np, name))

        raise AttributeError(name)

    def sum(self, a, axis=None, dtype=None):
        return rec_map_reduce_array_container(sum, partial(np.sum,
                                                           axis=axis,
                                                           dtype=dtype),
                                              a)

    def min(self, a, axis=None):
        return rec_map_reduce_array_container(
                partial(reduce, np.minimum), partial(np.amin, axis=axis), a)

    def max(self, a, axis=None):
        return rec_map_reduce_array_container(
                partial(reduce, np.maximum), partial(np.amax, axis=axis), a)

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
                lambda *args: np.stack(arrays=args, axis=axis),
                *arrays)

    def broadcast_to(self, array, shape):
        return rec_map_array_container(partial(np.broadcast_to, shape=shape), array)

    # {{{ relational operators

    def equal(self, x, y):
        return rec_multimap_array_container(np.equal, x, y)

    def not_equal(self, x, y):
        return rec_multimap_array_container(np.not_equal, x, y)

    def greater(self, x, y):
        return rec_multimap_array_container(np.greater, x, y)

    def greater_equal(self, x, y):
        return rec_multimap_array_container(np.greater_equal, x, y)

    def less(self, x, y):
        return rec_multimap_array_container(np.less, x, y)

    def less_equal(self, x, y):
        return rec_multimap_array_container(np.less_equal, x, y)

    # }}}

    def ravel(self, a, order="C"):
        return rec_map_array_container(partial(np.ravel, order=order), a)

    def vdot(self, x, y, dtype=None):
        if dtype is not None:
            raise NotImplementedError("only 'dtype=None' supported.")

        return rec_multimap_reduce_array_container(sum, np.vdot, x, y)

    def any(self, a):
        return rec_map_reduce_array_container(partial(reduce, np.logical_or),
                                              lambda subary: np.any(subary), a)

    def all(self, a):
        return rec_map_reduce_array_container(partial(reduce, np.logical_and),
                                              lambda subary: np.all(subary), a)

    def array_equal(self, a: ArrayOrContainer, b: ArrayOrContainer) -> Array:
        def rec_equal(x: ArrayOrContainer, y: ArrayOrContainer) -> np.ndarray:
            false_ary = np.array(False)
            true_ary = np.array(True)
            if type(x) is not type(y):
                return false_ary

            try:
                serialized_x = serialize_container(x)
                serialized_y = serialize_container(y)
            except NotAnArrayContainerError:
                assert isinstance(x, np.ndarray)
                assert isinstance(y, np.ndarray)
                return np.array(np.array_equal(x, y))
            else:
                if len(serialized_x) != len(serialized_y):
                    return false_ary
                return reduce(
                        np.logical_and,
                        [(true_ary if kx_i == ky_i else false_ary)
                            and rec_equal(x_i, y_i)
                            for (kx_i, x_i), (ky_i, y_i)
                            in zip(serialized_x, serialized_y)],
                        true_ary)

        result = rec_equal(a, b)

        return result

    def arange(self, *args, **kwargs):
        return np.arange(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        return np.linspace(*args, **kwargs)

    def zeros_like(self, ary):
        return rec_multimap_array_container(np.zeros_like, ary)

    def reshape(self, a, newshape, order="C"):
        return rec_map_array_container(
                lambda ary: ary.reshape(newshape, order=order),
                a)


# vim: fdm=marker
