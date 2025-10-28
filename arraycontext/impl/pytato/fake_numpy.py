from __future__ import annotations


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
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
from typing_extensions import override

import pytato as pt

from arraycontext.container import NotAnArrayContainerError, serialize_container
from arraycontext.container.traversal import (
    rec_map_array_container,
    rec_map_container,
    rec_map_reduce_array_container,
    rec_multimap_array_container,
)
from arraycontext.fake_numpy import BaseFakeNumpyLinalgNamespace
from arraycontext.loopy import LoopyBasedFakeNumpyNamespace
from arraycontext.typing import ArrayOrContainer, ArrayOrScalar, OrderCF, is_scalar_like


if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from numpy.typing import DTypeLike

    from pymbolic import Scalar

    from arraycontext.impl.pytato import _BasePytatoArrayContext
    from arraycontext.typing import Array, ArrayOrContainerOrScalar


class PytatoFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    # Everything is implemented in the base class for now.
    pass


class PytatoFakeNumpyNamespace(LoopyBasedFakeNumpyNamespace):
    """
    A :mod:`numpy` mimic for :class:`PytatoPyOpenCLArrayContext`.

    .. note::

        :mod:`pytato` does not define any memory layout for its arrays. See
        :ref:`Pytato docs <pytato:memory-layout>` for more on this.
    """
    _array_context: _BasePytatoArrayContext

    _pt_unary_funcs: Collection[str] = frozenset({
        "sin", "cos", "tan", "arcsin", "arccos", "arctan",
        "sinh", "cosh", "tanh", "exp", "log", "log10",
        "sqrt", "abs", "isnan", "real", "imag", "conj",
        "logical_not",
        })

    _pt_multi_ary_funcs: Collection[str] = frozenset({
        "arctan2", "equal", "greater", "greater_equal", "less", "less_equal",
        "not_equal", "minimum", "maximum", "where", "logical_and", "logical_or",
    })

    @override
    def _get_fake_numpy_linalg_namespace(self):
        return PytatoFakeNumpyLinalgNamespace(self._array_context)

    def __getattr__(self, name: str):
        if name in self._pt_unary_funcs:
            from functools import partial
            return partial(rec_map_array_container, getattr(pt, name))

        if name in self._pt_multi_ary_funcs:
            from functools import partial
            return partial(rec_multimap_array_container, getattr(pt, name))

        return super().__getattr__(name)

    # NOTE: the order of these follows the order in numpy docs
    # NOTE: when adding a function here, also add it to `array_context.rst` docs!

    # {{{ array creation routines

    @override
    def zeros(self, shape: int | tuple[int, ...], dtype: DTypeLike) -> Array:
        return pt.zeros(shape, dtype)

    @override
    def _full_like_array(self,
                ary: Array,
                fill_value: Scalar,
            ) -> Array:
        ...
        ary = cast("pt.Array", ary)
        return pt.full(ary.shape, fill_value, ary.dtype).copy(
                axes=ary.axes, tags=ary.tags)

    def arange(self, *args: Any, **kwargs: Any):
        return pt.arange(*args, **kwargs)

    # }}}

    # {{{ array manipulation routines

    @override
    def ravel(self, a: ArrayOrContainerOrScalar, order: OrderCF = "C"):
        """
        :arg order: A :class:`str` describing the order in which the elements
            must be traversed while flattening. Can be one of 'F', 'C', 'A' or
            'K'. Since, :mod:`pytato` arrays don't have a memory layout, if
            *order* is 'A' or 'K', the traversal order while flattening is
            undefined.
        """

        def _rec_ravel(a: ArrayOrScalar):
            if is_scalar_like(a):
                raise ValueError("cannot ravel scalars")

            a = cast("pt.Array", a)
            if order in "FC":
                return pt.reshape(a, (-1,), order=order)
            elif order in "AK":
                # flattening in a C-order
                # memory layout is assumed to be "C"
                return pt.reshape(a, (-1,), order="C")
            else:
                raise ValueError("`order` can be one of 'F', 'C', 'A' or 'K'. "
                                 f"(got {order})")

        return rec_map_container(_rec_ravel, a)

    def broadcast_to(self, array: ArrayOrContainerOrScalar, shape: tuple[int, ...]):
        def inner_bcast(ary: ArrayOrScalar) -> ArrayOrScalar:
            if is_scalar_like(ary):
                return ary
            else:
                assert isinstance(ary, pt.Array)
                return pt.broadcast_to(ary, shape)

        return rec_map_container(inner_bcast, array)

    def concatenate(self, arrays, axis=0):
        return rec_multimap_array_container(pt.concatenate, arrays, axis)

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
                lambda *args: pt.stack(arrays=args, axis=axis),
                *arrays)

    # }}}

    # {{{ logic functions

    def all(self, a, /):
        return rec_map_reduce_array_container(
                partial(reduce, pt.logical_and),
                lambda subary: pt.all(subary), a)

    def any(self, a, /):
        return rec_map_reduce_array_container(
                partial(reduce, pt.logical_or),
                lambda subary: pt.any(subary), a)

    @override
    def array_equal(self,
                a: ArrayOrContainerOrScalar,
                b: ArrayOrContainerOrScalar
            ) -> Array:
        actx = self._array_context

        # NOTE: not all backends support `bool` properly, so use `int8` instead
        true_ary = actx.from_numpy(np.int8(True))
        false_ary = actx.from_numpy(np.int8(False))

        def rec_equal(
                    x: ArrayOrContainerOrScalar,
                    y: ArrayOrContainerOrScalar
                ) -> Array:
            if type(x) is not type(y):
                return false_ary

            try:
                serialized_x = serialize_container(x)
                serialized_y = serialize_container(y)
            except NotAnArrayContainerError:
                assert isinstance(x, pt.Array)
                assert isinstance(y, pt.Array)

                if x.shape != y.shape:
                    return false_ary
                else:
                    return pt.all(cast("pt.Array", pt.equal(x, y)))
            else:
                if len(serialized_x) != len(serialized_y):
                    return false_ary

                return reduce(
                        cast("Callable[[Array, Array], Array]", pt.logical_and),
                        [(true_ary if kx_i == ky_i else false_ary)
                            and rec_equal(x_i, y_i)
                            for (kx_i, x_i), (ky_i, y_i)
                            in zip(serialized_x, serialized_y, strict=True)],
                        true_ary)

        return rec_equal(a, b)

    # }}}

    # {{{ mathematical functions

    @overload
    def sum(self,
                a: ArrayOrContainer,
                axis: int | tuple[int, ...] | None = None,
                dtype: DTypeLike = None,
            ) -> Array: ...
    @overload
    def sum(self,
                a: Scalar,
                axis: int | tuple[int, ...] | None = None,
                dtype: DTypeLike = None,
            ) -> Scalar: ...

    @override
    def sum(self,
                a: ArrayOrContainerOrScalar,
                axis: int | tuple[int, ...] | None = None,
                dtype: DTypeLike = None,
            ) -> ArrayOrScalar:
        def _pt_sum(ary):
            if dtype not in [ary.dtype, None]:
                raise NotImplementedError

            return pt.sum(ary, axis=axis)

        return rec_map_reduce_array_container(sum, _pt_sum, a)

    @overload
    def max(self,
                a: ArrayOrContainer,
                axis: int | tuple[int, ...] | None = None,
            ) -> Array: ...
    @overload
    def max(self,
                a: Scalar,
                axis: int | tuple[int, ...] | None = None,
            ) -> Scalar: ...

    @override
    def max(self,
                a: ArrayOrContainerOrScalar,
                axis: int | tuple[int, ...] | None = None,
            ) -> ArrayOrScalar:
        return rec_map_reduce_array_container(
                partial(reduce, pt.maximum), partial(pt.amax, axis=axis), a)

    amax = max  # pyright: ignore[reportAssignmentType, reportDeprecated]

    @overload
    def min(self,
                a: ArrayOrContainer,
                axis: int | tuple[int, ...] | None = None,
            ) -> Array: ...
    @overload
    def min(self,
                a: Scalar,
                axis: int | tuple[int, ...] | None = None,
            ) -> Scalar: ...

    @override
    def min(self,
                a: ArrayOrContainerOrScalar,
                axis: int | tuple[int, ...] | None = None,
            ) -> ArrayOrScalar:
        return rec_map_reduce_array_container(
                partial(reduce, pt.minimum), partial(pt.amin, axis=axis), a)

    amin = min  # pyright: ignore[reportDeprecated, reportAssignmentType]

    def absolute(self, a):
        return self.abs(a)

    def vdot(self, a: Array, b: Array):
        return rec_multimap_array_container(pt.vdot, a, b)

    # }}}
