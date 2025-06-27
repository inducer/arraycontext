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
from typing import TYPE_CHECKING, cast

import numpy as np
from typing_extensions import override

from arraycontext.container import NotAnArrayContainerError, serialize_container
from arraycontext.container.traversal import (
    rec_map_container,
    rec_map_reduce_array_container,
    rec_multimap_array_container,
    rec_multimap_reduce_array_container,
)
from arraycontext.context import OrderCF, is_scalar_like
from arraycontext.fake_numpy import (
    BaseFakeNumpyLinalgNamespace,
    BaseFakeNumpyNamespace,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike

    from pymbolic import Scalar

    from arraycontext.context import (
        Array,
        ArrayOrContainerOrScalar,
        ArrayOrScalar,
    )


class NumpyFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    # Everything is implemented in the base class for now.
    pass


_NUMPY_UFUNCS = frozenset({"concatenate", "reshape",
                 "ones_like", "where",
                 *BaseFakeNumpyNamespace._numpy_math_functions
                 })


class NumpyFakeNumpyNamespace(BaseFakeNumpyNamespace):
    """
    A :mod:`numpy` mimic for :class:`NumpyArrayContext`.
    """
    def _get_fake_numpy_linalg_namespace(self):
        return NumpyFakeNumpyLinalgNamespace(self._array_context)

    @override
    def zeros(self, shape: int | tuple[int, ...], dtype: DTypeLike) -> Array:
        return cast("Array", cast("object", np.zeros(shape, dtype)))

    @override
    def _full_like_array(self,
                ary: Array,
                fill_value: Scalar,
            ) -> Array:
        return cast("Array", cast("object", np.full_like(ary, fill_value)))

    def __getattr__(self, name: str):
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

    @override
    def min(self,
                a: ArrayOrContainerOrScalar,
                axis: int | tuple[int, ...] | None = None,
            ) -> ArrayOrScalar:
        return rec_map_reduce_array_container(
                partial(reduce, np.minimum), partial(np.amin, axis=axis), a)

    @override
    def max(self,
                a: ArrayOrContainerOrScalar,
                axis: int | tuple[int, ...] | None = None,
            ) -> ArrayOrScalar:
        return rec_map_reduce_array_container(
                partial(reduce, np.maximum), partial(np.amax, axis=axis), a)

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
                lambda *args: np.stack(arrays=args, axis=axis),
                *arrays)

    def broadcast_to(self, array: ArrayOrContainerOrScalar, shape: tuple[int, ...]):
        def inner_bcast(ary: ArrayOrScalar) -> ArrayOrScalar:
            if is_scalar_like(ary):
                return ary
            else:
                assert isinstance(ary, np.ndarray)
                return cast("Array", cast("object", np.broadcast_to(ary, shape)))

        return rec_map_container(inner_bcast, array)

    # {{{ relational operators

    @override
    def equal(self, x, y):
        return rec_multimap_array_container(np.equal, x, y)

    @override
    def not_equal(self, x, y):
        return rec_multimap_array_container(np.not_equal, x, y)

    @override
    def greater(self, x, y):
        return rec_multimap_array_container(np.greater, x, y)

    @override
    def greater_equal(self, x, y):
        return rec_multimap_array_container(np.greater_equal, x, y)

    @override
    def less(self, x, y):
        return rec_multimap_array_container(np.less, x, y)

    @override
    def less_equal(self, x, y):
        return rec_multimap_array_container(np.less_equal, x, y)

    @override
    def logical_or(self,
                x: ArrayOrContainerOrScalar,
                y: ArrayOrContainerOrScalar
            ) -> Array:
        return rec_multimap_array_container(np.logical_or, x, y)

    @override
    def logical_and(self,
                x: ArrayOrContainerOrScalar,
                y: ArrayOrContainerOrScalar
            ) -> Array:
        return rec_multimap_array_container(np.logical_and, x, y)

    @override
    def logical_not(self,
                x: ArrayOrContainerOrScalar
            ) -> ArrayOrContainerOrScalar:
        return rec_map_container(
                cast("Callable[[ArrayOrScalar], ArrayOrScalar]", np.logical_not), x)

    # }}}

    @override
    def ravel(self, a: ArrayOrContainerOrScalar, order: OrderCF = "C"):
        def inner_ravel(ary: ArrayOrScalar) -> ArrayOrScalar:
            if is_scalar_like(ary):
                return ary
            else:
                assert isinstance(ary, np.ndarray)
                return cast("Array", cast("object", np.ravel(ary, order)))

        return rec_map_container(inner_ravel, a)

    def vdot(self, x, y):
        return rec_multimap_reduce_array_container(sum, np.vdot, x, y)

    def any(self, a):
        return rec_map_reduce_array_container(partial(reduce, np.logical_or),
                                              lambda subary: np.any(subary), a)

    def all(self, a):
        return rec_map_reduce_array_container(partial(reduce, np.logical_and),
                                              lambda subary: np.all(subary), a)

    @override
    def array_equal(self,
                a: ArrayOrContainerOrScalar,
                b: ArrayOrContainerOrScalar
            ) -> Array:
        false_ary = np.array(False)
        true_ary = np.array(True)
        if type(a) is not type(b):
            return false_ary

        try:
            serialized_x = serialize_container(a)
            serialized_y = serialize_container(b)
        except NotAnArrayContainerError:
            assert isinstance(a, np.ndarray)
            assert isinstance(b, np.ndarray)
            return np.array(np.array_equal(a, b))
        else:
            if len(serialized_x) != len(serialized_y):
                return false_ary
            return np.logical_and.reduce(
                    [(true_ary if kx_i == ky_i else false_ary)
                        and cast("np.ndarray", self.array_equal(x_i, y_i))
                        for (kx_i, x_i), (ky_i, y_i)
                        in zip(serialized_x, serialized_y, strict=True)],
                    initial=true_ary)

    @override
    def arange(self, *args, **kwargs):
        return np.arange(*args, **kwargs)

    @override
    def linspace(self, *args, **kwargs):
        return np.linspace(*args, **kwargs)


# vim: fdm=marker
