"""
.. currentmodule:: arraycontext
.. autoclass:: PyOpenCLArrayContext
"""
__copyright__ = """
Copyright (C) 2020-1 University of Illinois Board of Trustees
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
import operator

import numpy as np

from arraycontext.fake_numpy import \
        BaseFakeNumpyNamespace, BaseFakeNumpyLinalgNamespace
from arraycontext.container import NotAnArrayContainerError, serialize_container
from arraycontext.container.traversal import (
        rec_map_array_container,
        rec_multimap_array_container,
        rec_map_reduce_array_container,
        rec_multimap_reduce_array_container,
        )

try:
    import pyopencl as cl  # noqa: F401
    import pyopencl.array as cl_array
except ImportError:
    pass


# {{{ fake numpy

class _NoValue:
    pass


class PyOpenCLFakeNumpyNamespace(BaseFakeNumpyNamespace):
    def _get_fake_numpy_linalg_namespace(self):
        return _PyOpenCLFakeNumpyLinalgNamespace(self._array_context)

    # {{{ comparisons

    # FIXME: This should be documentation, not a comment.
    # These are here mainly because some arrays may choose to interpret
    # equality comparison as a binary predicate of structural identity,
    # i.e. more like "are you two equal", and not like numpy semantics.
    # These operations provide access to numpy-style comparisons in that
    # case.

    def equal(self, x, y):
        return rec_multimap_array_container(operator.eq, x, y)

    def not_equal(self, x, y):
        return rec_multimap_array_container(operator.ne, x, y)

    def greater(self, x, y):
        return rec_multimap_array_container(operator.gt, x, y)

    def greater_equal(self, x, y):
        return rec_multimap_array_container(operator.ge, x, y)

    def less(self, x, y):
        return rec_multimap_array_container(operator.lt, x, y)

    def less_equal(self, x, y):
        return rec_multimap_array_container(operator.le, x, y)

    # }}}

    def ones_like(self, ary):
        def _ones_like(subary):
            ones = self._array_context.empty_like(subary)
            ones.fill(1)
            return ones

        return self._new_like(ary, _ones_like)

    def maximum(self, x, y):
        return rec_multimap_array_container(
                partial(cl_array.maximum, queue=self._array_context.queue),
                x, y)

    def minimum(self, x, y):
        return rec_multimap_array_container(
                partial(cl_array.minimum, queue=self._array_context.queue),
                x, y)

    def where(self, criterion, then, else_):
        def where_inner(inner_crit, inner_then, inner_else):
            if isinstance(inner_crit, bool):
                return inner_then if inner_crit else inner_else
            return cl_array.if_positive(inner_crit != 0, inner_then, inner_else,
                    queue=self._array_context.queue)

        return rec_multimap_array_container(where_inner, criterion, then, else_)

    @staticmethod
    def _map_reduce(reduce_func, map_func,
            ary, *,
            axis, dtype, initial):
        if isinstance(axis, int):
            axis = axis,

        def _reduce_func(partial_results):
            if initial is _NoValue:
                try:
                    return reduce(reduce_func, partial_results)
                except TypeError as exc:
                    if "empty sequence" in str(exc):
                        raise ValueError("zero-size reduction operation "
                                "without supplied 'initial' value")
                    else:
                        raise
            else:
                return reduce(reduce_func, partial_results, initial)

        def _map_func(ary):
            if dtype not in [None, ary.dtype]:
                raise NotImplementedError
            if axis not in [None, tuple(range(ary.ndim))]:
                raise NotImplementedError(
                    f"Mapping over '{axis}' axes not supported.")
            if initial is _NoValue:
                return map_func(ary)
            else:
                return map_func(ary, initial=initial)

        return rec_map_reduce_array_container(
                _reduce_func,
                _map_func,
                ary)

    # * appears where positional signature starts diverging from numpy
    def sum(self, a, axis=None, dtype=None, *, initial=0):
        queue = self._array_context.queue
        import operator
        result = self._map_reduce(
            operator.add,
            partial(cl_array.sum, queue=queue),
            a,
            axis=axis, dtype=dtype, initial=initial)

        if not self._array_context._force_device_scalars:
            result = result.get()[()]
        return result

    # * appears where positional signature starts diverging from numpy
    # Use _NoValue to indicate a lack of neutral element so that the caller can
    # use None as a neutral element
    def min(self, a, axis=None, *, initial=_NoValue):
        queue = self._array_context.queue
        result = self._map_reduce(
            partial(cl_array.minimum, queue=queue),
            partial(cl_array.min, queue=queue),
            a,
            axis=axis, dtype=None, initial=initial)

        if not self._array_context._force_device_scalars:
            result = result.get()[()]
        return result

    # * appears where positional signature starts diverging from numpy
    # Use _NoValue to indicate a lack of neutral element so that the caller can
    # use None as a neutral element
    def max(self, a, axis=None, *, initial=_NoValue):
        queue = self._array_context.queue
        result = self._map_reduce(
            partial(cl_array.maximum, queue=queue),
            partial(cl_array.max, queue=queue),
            a,
            axis=axis, dtype=None, initial=initial)

        if not self._array_context._force_device_scalars:
            result = result.get()[()]
        return result

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
                lambda *args: cl_array.stack(arrays=args, axis=axis,
                    queue=self._array_context.queue),
                *arrays)

    def reshape(self, a, newshape, order="C"):
        return rec_map_array_container(
                lambda ary: ary.reshape(newshape, order=order),
                a)

    def concatenate(self, arrays, axis=0):
        return cl_array.concatenate(
            arrays, axis,
            self._array_context.queue,
            self._array_context.allocator
        )

    def ravel(self, a, order="C"):
        def _rec_ravel(a):
            if order in "FC":
                return a.reshape(-1, order=order)
            elif order == "A":
                # TODO: upstream this to pyopencl.array
                if a.flags.f_contiguous:
                    return a.reshape(-1, order="F")
                elif a.flags.c_contiguous:
                    return a.reshape(-1, order="C")
                else:
                    raise ValueError("For `order='A'`, array should be either"
                                     " F-contiguous or C-contiguous.")
            elif order == "K":
                raise NotImplementedError("PyOpenCLArrayContext.np.ravel not "
                                          "implemented for 'order=K'")
            else:
                raise ValueError("`order` can be one of 'F', 'C', 'A' or 'K'. "
                                 f"(got {order})")

        return rec_map_array_container(_rec_ravel, a)

    def vdot(self, x, y, dtype=None):
        result = rec_multimap_reduce_array_container(
                sum,
                partial(cl_array.vdot, dtype=dtype, queue=self._array_context.queue),
                x, y)

        if not self._array_context._force_device_scalars:
            result = result.get()[()]
        return result

    def any(self, a):
        queue = self._array_context.queue
        result = rec_map_reduce_array_container(
                partial(reduce, partial(cl_array.maximum, queue=queue)),
                lambda subary: subary.any(queue=queue),
                a)

        if not self._array_context._force_device_scalars:
            result = result.get()[()]
        return result

    def all(self, a):
        queue = self._array_context.queue
        result = rec_map_reduce_array_container(
                partial(reduce, partial(cl_array.minimum, queue=queue)),
                lambda subary: subary.all(queue=queue),
                a)

        if not self._array_context._force_device_scalars:
            result = result.get()[()]
        return result

    def array_equal(self, a, b):
        actx = self._array_context
        queue = actx.queue

        # NOTE: pyopencl doesn't like `bool` much, so use `int8` instead
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
                    return (x == y).all()
            else:
                return reduce(
                        partial(cl_array.minimum, queue=queue),
                        [rec_equal(ix, iy)for (_, ix), (_, iy) in iterable]
                        )

        result = rec_equal(a, b)
        if not self._array_context._force_device_scalars:
            result = result.get()[()]

        return result

# }}}


# {{{ fake np.linalg

class _PyOpenCLFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    pass

# }}}


# vim: foldmethod=marker
