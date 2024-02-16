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

import cupy as cp  # type: ignore[import-untyped]  # pylint: disable=import-error

from arraycontext.container import is_array_container
from arraycontext.container.traversal import (
    multimap_reduce_array_container, rec_map_array_container,
    rec_map_reduce_array_container, rec_multimap_array_container,
    rec_multimap_reduce_array_container)
from arraycontext.fake_numpy import (
    BaseFakeNumpyLinalgNamespace, BaseFakeNumpyNamespace)


class CupyFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    # Everything is implemented in the base class for now.
    pass


_NUMPY_UFUNCS = {"abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
                 "sinh", "cosh", "tanh", "exp", "log", "log10", "isnan",
                 "sqrt", "concatenate", "transpose",
                 "ones_like", "maximum", "minimum", "where", "conj", "arctan2",
                 }


class CupyFakeNumpyNamespace(BaseFakeNumpyNamespace):
    """
    A :mod:`numpy` mimic for :class:`CupyArrayContext`.
    """
    def _get_fake_numpy_linalg_namespace(self):
        return CupyFakeNumpyLinalgNamespace(self._array_context)

    def __getattr__(self, name):

        if name in _NUMPY_UFUNCS:
            from functools import partial
            return partial(rec_multimap_array_container,
                           getattr(cp, name))

        raise NotImplementedError

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

    def vdot(self, x, y, dtype=None):
        if dtype is not None:
            raise NotImplementedError("only 'dtype=None' supported.")

        return rec_multimap_reduce_array_container(sum, cp.vdot, x, y)

    def any(self, a):
        return rec_map_reduce_array_container(partial(reduce, cp.logical_or),
                                              lambda subary: cp.any(subary), a)

    def all(self, a):
        return rec_map_reduce_array_container(partial(reduce, cp.logical_and),
                                              lambda subary: cp.all(subary), a)

    def array_equal(self, a, b):
        if type(a) is not type(b):
            return False
        elif not is_array_container(a):
            if a.shape != b.shape:
                return False
            else:
                return cp.all(cp.equal(a, b))
        else:
            try:
                return multimap_reduce_array_container(partial(reduce,
                                                           cp.logical_and),
                                                   self.array_equal, a, b)
            except TypeError:
                return True

    def zeros_like(self, ary):
        return rec_multimap_array_container(cp.zeros_like, ary)

    def reshape(self, a, newshape, order="C"):
        return rec_map_array_container(
                lambda ary: ary.reshape(newshape, order=order),
                a)

    def arange(self, *args, **kwargs):
        return cp.arange(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        return cp.linspace(*args, **kwargs)

# vim: fdm=marker
