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

from arraycontext.fake_numpy import (
        BaseFakeNumpyNamespace, BaseFakeNumpyLinalgNamespace,
        )
from arraycontext.container import NotAnArrayContainerError, serialize_container
from arraycontext.container.traversal import (
        rec_map_array_container,
        rec_multimap_array_container,
        rec_map_reduce_array_container,
        )
import pytato as pt


class PytatoFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    # Everything is implemented in the base class for now.
    pass


class _NoValue:
    pass


class PytatoFakeNumpyNamespace(BaseFakeNumpyNamespace):
    """
    A :mod:`numpy` mimic for :class:`PytatoPyOpenCLArrayContext`.

    .. note::

        :mod:`pytato` does not define any memory layout for its arrays. See
        :ref:`Pytato docs <pytato:memory-layout>` for more on this.
    """
    def _get_fake_numpy_linalg_namespace(self):
        return PytatoFakeNumpyLinalgNamespace(self._array_context)

    def __getattr__(self, name):

        pt_funcs = ["abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
                    "sinh", "cosh", "tanh", "exp", "log", "log10", "isnan",
                    "sqrt", "exp"]
        if name in pt_funcs:
            from functools import partial
            return partial(rec_map_array_container, getattr(pt, name))

        return super().__getattr__(name)

    def reshape(self, a, newshape, order="C"):
        return rec_map_array_container(
                lambda ary: pt.reshape(a, newshape, order=order),
                a)

    def transpose(self, a, axes=None):
        return rec_multimap_array_container(pt.transpose, a, axes)

    def concatenate(self, arrays, axis=0):
        return rec_multimap_array_container(pt.concatenate, arrays, axis)

    def ones_like(self, ary):
        def _ones_like(subary):
            return pt.ones(subary.shape, subary.dtype)

        return self._new_like(ary, _ones_like)

    def maximum(self, x, y):
        return rec_multimap_array_container(pt.maximum, x, y)

    def minimum(self, x, y):
        return rec_multimap_array_container(pt.minimum, x, y)

    def where(self, criterion, then, else_):
        return rec_multimap_array_container(pt.where, criterion, then, else_)

    @staticmethod
    def _reduce(container_binop, array_reduce,
            ary, *,
            axis, dtype, initial):
        def container_reduce(ctr):
            if initial is _NoValue:
                try:
                    return reduce(container_binop, ctr)
                except TypeError as exc:
                    assert "empty sequence" in str(exc)
                    raise ValueError("zero-size reduction operation "
                            "without supplied 'initial' value")
            else:
                return reduce(container_binop, ctr, initial)

        def actual_array_reduce(ary):
            if dtype not in [ary.dtype, None]:
                raise NotImplementedError

            if initial is _NoValue:
                return array_reduce(ary, axis=axis)
            else:
                return array_reduce(ary, axis=axis, initial=initial)

        return rec_map_reduce_array_container(
                container_reduce,
                actual_array_reduce,
                ary)

    # * appears where positional signature starts diverging from numpy
    def sum(self, a, axis=None, dtype=None, *, initial=0):
        import operator
        return self._reduce(operator.add, pt.sum, a,
                axis=axis, dtype=dtype, initial=initial)

    # * appears where positional signature starts diverging from numpy
    def min(self, a, axis=None, *, initial=_NoValue):
        return self._reduce(pt.minimum, pt.amin, a,
                axis=axis, dtype=None, initial=initial)

    # * appears where positional signature starts diverging from numpy
    def max(self, a, axis=None, *, initial=_NoValue):
        return self._reduce(pt.maximum, pt.amax, a,
                axis=axis, dtype=None, initial=initial)

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
                lambda *args: pt.stack(arrays=args, axis=axis),
                *arrays)

    def broadcast_to(self, array, shape):
        return rec_map_array_container(partial(pt.broadcast_to, shape=shape), array)

    # {{{ relational operators

    def equal(self, x, y):
        return rec_multimap_array_container(pt.equal, x, y)

    def not_equal(self, x, y):
        return rec_multimap_array_container(pt.not_equal, x, y)

    def greater(self, x, y):
        return rec_multimap_array_container(pt.greater, x, y)

    def greater_equal(self, x, y):
        return rec_multimap_array_container(pt.greater_equal, x, y)

    def less(self, x, y):
        return rec_multimap_array_container(pt.less, x, y)

    def less_equal(self, x, y):
        return rec_multimap_array_container(pt.less_equal, x, y)

    def conj(self, x):
        return rec_multimap_array_container(pt.conj, x)

    def arctan2(self, y, x):
        return rec_multimap_array_container(pt.arctan2, y, x)

    def ravel(self, a, order="C"):
        """
        :arg order: A :class:`str` describing the order in which the elements
            must be traversed while flattening. Can be one of 'F', 'C', 'A' or
            'K'. Since, :mod:`pytato` arrays don't have a memory layout, if
            *order* is 'A' or 'K', the traversal order while flattening is
            undefined.
        """

        def _rec_ravel(a):
            if order in "FC":
                return pt.reshape(a, (-1,), order=order)
            elif order in "AK":
                # flattening in a C-order
                # memory layout is assumed to be "C"
                return pt.reshape(a, (-1,), order="C")
            else:
                raise ValueError("`order` can be one of 'F', 'C', 'A' or 'K'. "
                                 f"(got {order})")

        return rec_map_array_container(_rec_ravel, a)

    def any(self, a):
        return rec_map_reduce_array_container(
                partial(reduce, pt.logical_or),
                lambda subary: pt.any(subary), a)

    def all(self, a):
        return rec_map_reduce_array_container(
                partial(reduce, pt.logical_and),
                lambda subary: pt.all(subary), a)

    def array_equal(self, a, b):
        actx = self._array_context

        # NOTE: not all backends support `bool` properly, so use `int8` instead
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
                    return pt.all(pt.equal(x, y))
            else:
                return reduce(
                        pt.logical_and,
                        [rec_equal(ix, iy) for (_, ix), (_, iy) in iterable]
                        )

        return rec_equal(a, b)

    # }}}
