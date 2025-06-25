"""
.. currentmodule:: arraycontext
.. autoclass:: PyOpenCLArrayContext
"""
from __future__ import annotations


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

import operator
from functools import partial, reduce
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from typing_extensions import override

import pyopencl.array as cl_array

from arraycontext.container import NotAnArrayContainerError, serialize_container
from arraycontext.container.traversal import (
    rec_map_container,
    rec_map_reduce_array_container,
    rec_multimap_array_container,
    rec_multimap_reduce_array_container,
)
from arraycontext.context import OrderCF, is_scalar_like
from arraycontext.fake_numpy import BaseFakeNumpyLinalgNamespace
from arraycontext.impl.pyopencl.taggable_cl_array import TaggableCLArray
from arraycontext.loopy import LoopyBasedFakeNumpyNamespace


if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from pymbolic import Scalar
    from pytools.tag import Tag

    from arraycontext.context import (
        Array,
        ArrayOrContainerOrScalar,
        ArrayOrScalar,
    )
    from arraycontext.impl.pyopencl import PyOpenCLArrayContext


# {{{ fake numpy

class PyOpenCLFakeNumpyNamespace(LoopyBasedFakeNumpyNamespace):
    _array_context: PyOpenCLArrayContext

    @override
    def _get_fake_numpy_linalg_namespace(self):
        return _PyOpenCLFakeNumpyLinalgNamespace(self._array_context)

    # NOTE: the order of these follows the order in numpy docs
    # NOTE: when adding a function here, also add it to `array_context.rst` docs!

    # {{{ array creation routines

    @override
    def zeros(self, shape: int | tuple[int, ...], dtype: DTypeLike) -> Array:
        import arraycontext.impl.pyopencl.taggable_cl_array as tga
        return tga.zeros(self._array_context.queue, shape, dtype,
                         allocator=self._array_context.allocator)

    def empty_like(self, ary):
        from warnings import warn
        warn(f"{type(self._array_context).__name__}.np.empty_like is "
            "deprecated and will stop working in 2023. Prefer actx.np.zeros_like "
            "instead.",
            DeprecationWarning, stacklevel=2)

        import arraycontext.impl.pyopencl.taggable_cl_array as tga
        actx = self._array_context

        def _empty_like(array):
            return tga.empty(actx.queue, array.shape, array.dtype,
                allocator=actx.allocator, axes=array.axes, tags=array.tags)

        return actx._rec_map_container(_empty_like, ary)

    @override
    def _full_like_array(self,
                ary: Array,
                fill_value: Scalar,
            ) -> Array:
        assert isinstance(ary, cl_array.Array)

        if isinstance(ary, TaggableCLArray):
            axes = ary.axes
            tags: frozenset[Tag] = ary.tags
        else:
            warn(f"{self._array_context.__class__.__name__} received a "
                 f"{ary.__class__.__qualname__}, "
                 "not a TaggableCLArray. This is deprecated and will stop working "
                 "in 2026.", DeprecationWarning, stacklevel=3)

            axes = None
            tags = frozenset()

        import arraycontext.impl.pyopencl.taggable_cl_array as tga
        actx = self._array_context

        filled = tga.empty(
            actx.queue, ary.shape, ary.dtype,
            allocator=actx.allocator, axes=axes, tags=tags)
        filled.fill(fill_value)

        return filled

    def copy(self, ary):
        def _copy(subary):
            return subary.copy(queue=self._array_context.queue)

        return self._array_context._rec_map_container(_copy, ary)

    def arange(self, *args, **kwargs):
        return cl_array.arange(self._array_context.queue, *args, **kwargs)

    # }}}

    # {{{ array manipulation routines

    @override
    def ravel(self,
                a: ArrayOrContainerOrScalar,
                order: OrderCF = "C"
            ) -> ArrayOrContainerOrScalar:
        def _rec_ravel(a: ArrayOrScalar) -> Array:
            if is_scalar_like(a):
                raise ValueError("cannot ravel scalars")
            if order in "FC":
                return a.reshape(-1, order=order)
            elif order == "A":
                from warnings import warn
                warn('order=="A" is deprecated, use one of "C", "F" instead',
                     DeprecationWarning, stacklevel=2)
                if a.flags.f_contiguous:
                    return a.reshape(-1, order="F")
                elif a.flags.c_contiguous:
                    return a.reshape(-1, order="C")
                else:
                    raise ValueError("For `order='A'`, array should be either"
                                     " F-contiguous or C-contiguous.")
            else:
                raise ValueError(f"`order` can be one of 'F', 'C'. (got {order})")

        return rec_map_container(_rec_ravel, a)

    def concatenate(self, arrays, axis=0):
        return cl_array.concatenate(
            arrays, axis,
            self._array_context.queue,
            self._array_context.allocator
        )

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
                lambda *args: cl_array.stack(arrays=args, axis=axis,
                    queue=self._array_context.queue),
                *arrays)

    # }}}

    # {{{ linear algebra

    def vdot(self, x, y, dtype=None):
        return rec_multimap_reduce_array_container(
                sum,
                partial(cl_array.vdot, dtype=dtype, queue=self._array_context.queue),
                x, y)

    # }}}

    # {{{ logic functions

    def all(self, a):
        queue = self._array_context.queue

        def _all(ary):
            if np.isscalar(ary):
                return np.int8(all([ary]))
            return ary.all(queue=queue)

        return rec_map_reduce_array_container(
                partial(reduce, partial(cl_array.minimum, queue=queue)),
                _all,
                a)

    def any(self, a):
        queue = self._array_context.queue

        def _any(ary):
            if np.isscalar(ary):
                return np.int8(any([ary]))
            return ary.any(queue=queue)

        return rec_map_reduce_array_container(
                partial(reduce, partial(cl_array.maximum, queue=queue)),
                _any,
                a)

    def array_equal(self,
                a: ArrayOrContainerOrScalar,
                b: ArrayOrContainerOrScalar
            ) -> Array:
        actx = self._array_context
        queue = actx.queue

        # NOTE: pyopencl doesn't like `bool` much, so use `int8` instead
        true_ary = actx.from_numpy(np.int8(True))
        false_ary = actx.from_numpy(np.int8(False))

        def rec_equal(
                    x: ArrayOrContainerOrScalar,
                    y: ArrayOrContainerOrScalar,
                ) -> cl_array.Array:
            if type(x) is not type(y):
                return false_ary

            try:
                serialized_x = serialize_container(x)
                serialized_y = serialize_container(y)
            except NotAnArrayContainerError:
                assert isinstance(x, cl_array.Array)
                assert isinstance(y, cl_array.Array)

                if x.shape != y.shape:
                    return false_ary
                else:
                    return (x == y).all()
            else:
                if len(serialized_x) != len(serialized_y):
                    return false_ary

                return reduce(
                        partial(cl_array.minimum, queue=queue),
                        [(true_ary if kx_i == ky_i else false_ary)
                            and rec_equal(x_i, y_i)
                            for (kx_i, x_i), (ky_i, y_i)
                            in zip(serialized_x, serialized_y, strict=True)],
                        true_ary)

        return rec_equal(a, b)

    @override
    def greater(self,
            x: ArrayOrContainerOrScalar,
            y: ArrayOrContainerOrScalar
        ) -> Array:
        return rec_multimap_array_container(operator.gt, x, y)

    @override
    def greater_equal(self,
                    x: ArrayOrContainerOrScalar,
                    y: ArrayOrContainerOrScalar
                ) -> Array:
        return rec_multimap_array_container(operator.ge, x, y)

    @override
    def less(self,
            x: ArrayOrContainerOrScalar,
            y: ArrayOrContainerOrScalar
        ) -> Array:
        return rec_multimap_array_container(operator.lt, x, y)

    @override
    def less_equal(self,
                x: ArrayOrContainerOrScalar,
                y: ArrayOrContainerOrScalar
            ) -> Array:
        return rec_multimap_array_container(operator.le, x, y)

    @override
    def equal(self,
            x: ArrayOrContainerOrScalar,
            y: ArrayOrContainerOrScalar
        ) -> Array:
        return rec_multimap_array_container(operator.eq, x, y)

    @override
    def not_equal(self,
                x: ArrayOrContainerOrScalar,
                y: ArrayOrContainerOrScalar
            ) -> Array:
        return rec_multimap_array_container(operator.ne, x, y)

    @override
    def logical_or(self,
                x: ArrayOrContainerOrScalar,
                y: ArrayOrContainerOrScalar
            ) -> Array:
        return rec_multimap_array_container(cl_array.logical_or, x, y)

    @override
    def logical_and(self,
                x: ArrayOrContainerOrScalar,
                y: ArrayOrContainerOrScalar
            ) -> Array:
        return rec_multimap_array_container(cl_array.logical_and, x, y)

    @override
    def logical_not(self,
                x: ArrayOrContainerOrScalar
            ) -> ArrayOrContainerOrScalar:

        def inner(ary: ArrayOrScalar) -> ArrayOrScalar:
            if is_scalar_like(ary):
                return ary
            else:
                assert isinstance(ary, cl_array.Array)
                return cl_array.logical_not(ary)

        return rec_map_container(inner, x)

    # }}}

    # {{{ mathematical functions

    def sum(self, a, axis=None, dtype=None):
        if isinstance(axis, int):
            axis = axis,

        def _rec_sum(ary):
            if axis not in [None, tuple(range(ary.ndim))]:
                raise NotImplementedError(f"Sum over '{axis}' axes not supported.")

            return cl_array.sum(ary, dtype=dtype, queue=self._array_context.queue)

        return rec_map_reduce_array_container(sum, _rec_sum, a)

    def maximum(self, x, y):
        return rec_multimap_array_container(
                partial(cl_array.maximum, queue=self._array_context.queue),
                x, y)

    @override
    def max(self,
                a: ArrayOrContainerOrScalar,
                axis: int | tuple[int, ...] | None = None,
            ) -> ArrayOrScalar:

        queue = self._array_context.queue

        if isinstance(axis, int):
            axis = axis,

        def _rec_max(ary):
            if axis not in [None, tuple(range(ary.ndim))]:
                raise NotImplementedError(f"Max. over '{axis}' axes not supported.")
            return cl_array.max(ary, queue=queue)

        return rec_map_reduce_array_container(
                partial(reduce, partial(cl_array.maximum, queue=queue)),
                _rec_max,
                a)

    amax = max

    def minimum(self, x, y):
        return rec_multimap_array_container(
                partial(cl_array.minimum, queue=self._array_context.queue),
                x, y)

    @override
    def min(self,
                a: ArrayOrContainerOrScalar,
                axis: int | tuple[int, ...] | None = None,
            ) -> ArrayOrScalar:
        queue = self._array_context.queue

        if isinstance(axis, int):
            axis = axis,

        def _rec_min(ary):
            if axis not in [None, tuple(range(ary.ndim))]:
                raise NotImplementedError(f"Min. over '{axis}' axes not supported.")
            return cl_array.min(ary, queue=queue)

        return rec_map_reduce_array_container(
                partial(reduce, partial(cl_array.minimum, queue=queue)),
                _rec_min,
                a)

    amin = min

    def absolute(self, a):
        return self.abs(a)

    # }}}

    # {{{ sorting, searching, and counting

    def where(self, criterion, then, else_):
        def where_inner(inner_crit, inner_then, inner_else):
            if isinstance(inner_crit, bool | np.bool_):
                return inner_then if inner_crit else inner_else
            return cl_array.if_positive(inner_crit != 0, inner_then, inner_else,
                    queue=self._array_context.queue)

        return rec_multimap_array_container(where_inner, criterion, then, else_)

    # }}}

# }}}


# {{{ fake np.linalg

class _PyOpenCLFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    pass

# }}}


# vim: foldmethod=marker
