# pyright: reportUnusedParameter=none

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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np

from arraycontext.container import (
    NotAnArrayContainerError,
    is_array_container,
    serialize_container,
)
from arraycontext.container.traversal import rec_map_container
from arraycontext.typing import ArrayOrContainer, ArrayOrContainerT, is_scalar_like


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import DTypeLike, NDArray

    from pymbolic import Scalar

    from arraycontext.context import ArrayContext
    from arraycontext.typing import (
        Array,
        ArrayOrContainerOrScalar,
        ArrayOrContainerOrScalarT,
        ArrayOrScalar,
        OrderCF,
    )


# {{{ BaseFakeNumpyNamespace

@dataclass(frozen=True)
class BaseFakeNumpyNamespace(ABC):
    _array_context: ArrayContext
    linalg: BaseFakeNumpyLinalgNamespace

    def __init__(self, array_context: ArrayContext):
        object.__setattr__(self, "_array_context", array_context)
        object.__setattr__(self, "linalg", self._get_fake_numpy_linalg_namespace())

    def _get_fake_numpy_linalg_namespace(self):
        return BaseFakeNumpyLinalgNamespace(self._array_context)

    _numpy_math_functions = frozenset({
        # https://numpy.org/doc/stable/reference/routines.math.html

        # FIXME: Heads up: not all of these are supported yet.
        # But I felt it was important to only dispatch actually existing
        # numpy functions to loopy.

        # Trigonometric functions
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "hypot", "arctan2",
        "degrees", "radians", "unwrap", "deg2rad", "rad2deg",

        # Hyperbolic functions
        "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",

        # Rounding
        "around", "round_", "rint", "fix", "floor", "ceil", "trunc",

        # Sums, products, differences

        # FIXME: Many of These are reductions or scans.
        # "prod", "sum", "nanprod", "nansum", "cumprod", "cumsum", "nancumprod",
        # "nancumsum", "diff", "ediff1d", "gradient", "cross", "trapz",

        # Exponents and logarithms
        "exp", "expm1", "exp2", "log", "log10", "log2", "log1p", "logaddexp",
        "logaddexp2",

        # Other special functions
        "i0", "sinc",

        # Floating point routines
        "signbit", "copysign", "frexp", "ldexp", "nextafter", "spacing",
        # Rational routines
        "lcm", "gcd",

        # Arithmetic operations
        "add", "reciprocal", "positive", "negative", "multiply", "divide", "power",
        "subtract", "true_divide", "floor_divide", "float_power", "fmod", "mod",
        "modf", "remainder", "divmod",

        # Handling complex numbers
        "angle", "real", "imag",
        # Implemented below:
        # "conj", "conjugate",

        # Miscellaneous
        "convolve", "clip", "sqrt", "cbrt", "square", "absolute", "abs", "fabs",
        "sign", "heaviside", "maximum", "fmax", "nan_to_num", "isnan", "minimum",
        "fmin",

        # FIXME:
        # "interp",
        })

    @abstractmethod
    def zeros(self, shape: int | tuple[int, ...], dtype: DTypeLike) -> Array:
        ...

    def zeros_like(self, ary: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        return self.full_like(ary, 0)

    @abstractmethod
    def _full_like_array(self,
                ary: Array,
                fill_value: Scalar,
            ) -> Array:
        ...

    def full_like(self,
                ary: ArrayOrContainerOrScalarT,
                fill_value: Scalar,
            ) -> ArrayOrContainerOrScalarT:
        def _zeros_like(array: ArrayOrScalar) -> ArrayOrScalar:
            if is_scalar_like(array):
                return fill_value
            else:
                return self._full_like_array(array, fill_value)

        return cast("ArrayOrContainerOrScalarT", rec_map_container(_zeros_like, ary))

    def ones_like(self, ary: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        return self.full_like(ary, 1)

    def conjugate(self, x: ArrayOrContainerOrScalar):
        # NOTE: conjugate distributes over object arrays, but it looks for a
        # `conjugate` ufunc, while some implementations only have the shorter
        # `conj` (e.g. cl.array.Array), so this should work for everybody.
        return rec_map_container(lambda obj: cast("Array", obj).conj(), x)

    def conj(self, x: ArrayOrContainerOrScalar):
        # NOTE: conjugate distributes over object arrays, but it looks for a
        # `conjugate` ufunc, while some implementations only have the shorter
        # `conj` (e.g. cl.array.Array), so this should work for everybody.
        return rec_map_container(lambda obj: cast("Array", obj).conj(), x)

    # {{{ linspace

    # based on
    # https://github.com/numpy/numpy/blob/v1.25.0/numpy/core/function_base.py#L24-L182

    @overload
    def linspace(self,
                start: NDArray[Any] | Scalar,
                stop: NDArray[Any] | Scalar,
                num: int = 50,
                *, endpoint: bool = True,
                retstep: Literal[False] = False,
                dtype: DTypeLike = None,
                axis: int = 0
            ) -> Array: ...

    @overload
    def linspace(self,
                start: NDArray[Any] | Scalar,
                stop: NDArray[Any] | Scalar,
                num: int = 50,
                *, endpoint: bool = True,
                retstep: Literal[True],
                dtype: DTypeLike = None,
                axis: int = 0
            ) -> tuple[Array, NDArray[Any] | float] | Array: ...

    def linspace(self,
                start: NDArray[Any] | Scalar,
                stop: NDArray[Any] | Scalar,
                num: int = 50,
                *, endpoint: bool = True,
                retstep: bool = False,
                dtype: DTypeLike = None,
                axis: int = 0
            ) -> tuple[Array, NDArray[Any] | float] | Array:
        num = operator.index(num)
        if num < 0:
            raise ValueError(f"Number of samples, {num}, must be non-negative.")
        div = (num - 1) if endpoint else num

        # Convert float/complex array scalars to float, gh-3504
        # and make sure one can use variables that have an __array_interface__,
        # gh-6634

        if isinstance(start, self._array_context.array_types):
            raise NotImplementedError("start as an actx array")
        if isinstance(stop, self._array_context.array_types):
            raise NotImplementedError("stop as an actx array")

        start = np.array(start) * 1.0
        stop = np.array(stop) * 1.0

        dt = np.result_type(start, stop, float(num))
        if dtype is None:
            dtype = dt
            integer_dtype = False
        else:
            integer_dtype = np.issubdtype(dtype, np.integer)

        delta = stop - start

        y = self.arange(0, num, dtype=dt).reshape(-1, *((1,) * delta.ndim))

        if div > 0:
            step = delta / div
            # any_step_zero = _nx.asanyarray(step == 0).any()
            any_step_zero = self._array_context.to_numpy(step == 0).any()
            if any_step_zero:
                delta_actx = self._array_context.from_numpy(delta)

                # Special handling for denormal numbers, gh-5437
                y = y / div
                y = y * delta_actx
            else:
                step_actx = self._array_context.from_numpy(step)
                y = y * step_actx
        else:
            delta_actx = self._array_context.from_numpy(delta)
            # sequences with 0 items or 1 item with endpoint=True (i.e. div <= 0)
            # have an undefined step
            step = np.nan
            # Multiply with delta to allow possible override of output class.
            y = y * delta_actx

        y += self._array_context.from_numpy(start)

        # FIXME reenable, without in-place ops
        # if endpoint and num > 1:
        #     y[-1, ...] = stop

        if axis != 0:
            # y = _nx.moveaxis(y, 0, axis)
            raise NotImplementedError("axis != 0")

        if integer_dtype:
            y = self.floor(y)

        if retstep:
            return y.astype(dtype), step
        else:
            return y.astype(dtype)

    # }}}

    def arange(self, *args: Any, **kwargs: Any) -> Array:
        raise NotImplementedError

    def reshape(self,
                a: ArrayOrContainer,
                /, shape: tuple[int, ...],
                order: OrderCF = "C"):
        def inner(a: ArrayOrScalar) -> Array:
            if is_scalar_like(a):
                raise ValueError("reshape not meaningful for scalars")

            return a.reshape(shape, order=order)

        return rec_map_container(inner, a)

    def transpose(self,
                a: ArrayOrContainer,
                /, axes: tuple[int, ...],
                ):
        def inner(a: ArrayOrScalar) -> ArrayOrScalar:
            if is_scalar_like(a):
                return a

            return a.transpose(axes)

        return rec_map_container(inner, a)

    if TYPE_CHECKING:
        # These at least pin down the type signatures. We cannot use abstract methods
        # here, because some of these are implemented via __getattr__ hacking in
        # subclasses. Defining them as abstract methods would define them
        # as attributes, making __getattr__ fail to retrieve the intended function.

        def broadcast_to(self,
                array: ArrayOrContainerOrScalar,
                shape: tuple[int, ...]
            ) -> ArrayOrContainerOrScalar: ...

        def concatenate(self,
                    arrays: Sequence[ArrayOrContainerT],
                    axis: int = 0
                ) -> ArrayOrContainerT: ...

        def stack(self,
                    arrays: Sequence[ArrayOrContainerT],
                    axis: int = 0
                ) -> ArrayOrContainerT: ...

        def ravel(self,
                    a: ArrayOrContainerOrScalarT,
                    order: OrderCF = "C"
                ) -> ArrayOrContainerOrScalarT: ...

        def array_equal(self,
                    a: ArrayOrContainerOrScalar,
                    b: ArrayOrContainerOrScalar
                ) -> Array: ...

        def sqrt(self,
                    a: ArrayOrContainerOrScalarT,
                ) -> ArrayOrContainerOrScalarT: ...

        def abs(self,
                    a: ArrayOrContainerOrScalarT,
                ) -> ArrayOrContainerOrScalarT: ...

        def sin(self,
                    a: ArrayOrContainerOrScalarT,
                ) -> ArrayOrContainerOrScalarT: ...

        def cos(self,
                    a: ArrayOrContainerOrScalarT,
                ) -> ArrayOrContainerOrScalarT: ...

        def floor(self,
                    a: ArrayOrContainerOrScalarT,
                ) -> ArrayOrContainerOrScalarT: ...

        def ceil(self,
                    a: ArrayOrContainerOrScalarT,
                ) -> ArrayOrContainerOrScalarT: ...

        # {{{ binary/ternary ufuncs

        # FIXME: These are more restrictive than necessary, but they'll do the job
        # for now.

        def minimum(self,
                    a: ArrayOrContainerOrScalarT,
                    b: ArrayOrContainerOrScalarT,
                    /,
                ) -> ArrayOrContainerOrScalarT: ...

        def maximum(self,
                    a: ArrayOrContainerOrScalarT,
                    b: ArrayOrContainerOrScalarT,
                    /,
                ) -> ArrayOrContainerOrScalarT: ...

        def atan2(self,
                    a: ArrayOrContainerOrScalarT,
                    b: ArrayOrContainerOrScalarT,
                    /,
                ) -> ArrayOrContainerOrScalarT: ...

        def where(self,
                    condition: ArrayOrContainerOrScalarT,
                    x: ArrayOrContainerOrScalar,
                    y: ArrayOrContainerOrScalar,
                    /,
                ) -> ArrayOrContainerOrScalarT: ...

        # }}}

        # {{{ reductions

        def sum(self,
                    a: ArrayOrContainerOrScalar,
                    axis: int | tuple[int, ...] | None = None,
                    dtype: DTypeLike = None,
                ) -> ArrayOrScalar: ...

        def max(self,
                    a: ArrayOrContainerOrScalar,
                    axis: int | tuple[int, ...] | None = None,
                ) -> ArrayOrScalar: ...

        def min(self,
                    a: ArrayOrContainerOrScalar,
                    axis: int | tuple[int, ...] | None = None,
                ) -> ArrayOrScalar: ...

        def amax(self,
                    a: ArrayOrContainerOrScalar,
                    axis: int | tuple[int, ...] | None = None,
                ) -> ArrayOrScalar: ...

        def amin(self,
                    a: ArrayOrContainerOrScalar,
                    axis: int | tuple[int, ...] | None = None,
                ) -> ArrayOrScalar: ...

        def any(self,
                    a: ArrayOrContainerOrScalar,
                ) -> ArrayOrScalar: ...

        def all(self,
                    a: ArrayOrContainerOrScalar,
                ) -> ArrayOrScalar: ...

        # }}}

        # FIXME: This should be documentation, not a comment.
        # These are here mainly because some arrays may choose to interpret
        # equality comparison as a binary predicate of structural identity,
        # i.e. more like "are you two equal", and not like numpy semantics.
        # These operations provide access to numpy-style comparisons in that
        # case.

        def greater(
                    self, x: ArrayOrContainerOrScalar, y: ArrayOrContainerOrScalar
                ) -> ArrayOrContainerOrScalar:
            ...

        def greater_equal(
                          self, x: ArrayOrContainerOrScalar, y: ArrayOrContainerOrScalar
                      ) -> ArrayOrContainerOrScalar:
            ...

        def less(
                 self, x: ArrayOrContainerOrScalar, y: ArrayOrContainerOrScalar
             ) -> ArrayOrContainerOrScalar:
            ...

        def less_equal(
                       self, x: ArrayOrContainerOrScalar, y: ArrayOrContainerOrScalar
                   ) -> ArrayOrContainerOrScalar:
            ...

        def equal(
                  self, x: ArrayOrContainerOrScalar, y: ArrayOrContainerOrScalar
              ) -> ArrayOrContainerOrScalar:
            ...

        def not_equal(
                      self, x: ArrayOrContainerOrScalar, y: ArrayOrContainerOrScalar
                  ) -> ArrayOrContainerOrScalar:
            ...

        def logical_or(
                       self, x: ArrayOrContainerOrScalar, y: ArrayOrContainerOrScalar
                   ) -> ArrayOrContainerOrScalar:
            ...

        def logical_and(
                        self, x: ArrayOrContainerOrScalar, y: ArrayOrContainerOrScalar
                    ) -> ArrayOrContainerOrScalar:
            ...

        def logical_not(
                        self, x: ArrayOrContainerOrScalar
                    ) -> ArrayOrContainerOrScalar:
            ...

# }}}


# {{{ BaseFakeNumpyLinalgNamespace

def _reduce_norm(actx: ArrayContext, arys: Iterable[ArrayOrScalar], ord: float | None):
    from functools import reduce
    from numbers import Number

    if ord is None:
        ord = 2

    # NOTE: these are ordered by an expected usage frequency
    if ord == 2:
        return actx.np.sqrt(sum(subary*subary for subary in arys))
    elif ord == np.inf:
        return reduce(actx.np.maximum, arys)
    elif ord == -np.inf:
        return reduce(actx.np.minimum, arys)
    elif isinstance(ord, Number) and ord > 0:
        return sum(subary**ord for subary in arys)**(1/ord)
    else:
        raise NotImplementedError(f"unsupported value of 'ord': {ord}")


class BaseFakeNumpyLinalgNamespace:
    _array_context: ArrayContext

    def __init__(self, array_context: ArrayContext):
        self._array_context = array_context

    def norm(self,
                 ary: ArrayOrContainerOrScalar,
                 ord: float | None = None
             ) -> ArrayOrScalar:
        if is_scalar_like(ary):
            return abs(ary)

        actx = self._array_context

        try:
            iterable = serialize_container(ary)
        except NotAnArrayContainerError:
            pass
        else:
            if TYPE_CHECKING:
                assert is_array_container(ary)

            return _reduce_norm(actx, [
                self.norm(subary, ord=ord) for _, subary in iterable
                ], ord=ord)

        if TYPE_CHECKING:
            assert not is_array_container(ary)

        if ord is None:
            return self.norm(actx.np.ravel(ary, order="C"), 2)

        if len(ary.shape) != 1:
            raise NotImplementedError("only vector norms are implemented")

        if ary.size == 0:
            return ary.dtype.type(0)

        from numbers import Number
        if ord == 2:
            return actx.np.sqrt(actx.np.sum(abs(ary)**2))
        if ord == np.inf:
            return actx.np.max(abs(ary))
        elif ord == -np.inf:
            return actx.np.min(abs(ary))
        elif isinstance(ord, Number) and ord > 0:
            return actx.np.sum(abs(ary)**ord)**(1/ord)
        else:
            raise NotImplementedError(f"unsupported value of 'ord': {ord}")

# }}}


# vim: foldmethod=marker
