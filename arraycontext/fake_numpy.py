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
from typing import Any

import numpy as np

from arraycontext.container import NotAnArrayContainerError, serialize_container
from arraycontext.container.traversal import rec_map_array_container


# {{{ BaseFakeNumpyNamespace

class BaseFakeNumpyNamespace(ABC):
    def __init__(self, array_context):
        self._array_context = array_context
        self.linalg = self._get_fake_numpy_linalg_namespace()

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
    def zeros(self, shape, dtype):
        ...

    @abstractmethod
    def zeros_like(self, ary):
        ...

    def conjugate(self, x):
        # NOTE: conjugate distributes over object arrays, but it looks for a
        # `conjugate` ufunc, while some implementations only have the shorter
        # `conj` (e.g. cl.array.Array), so this should work for everybody.
        return rec_map_array_container(lambda obj: obj.conj(), x)

    conj = conjugate

    # {{{ linspace

    # based on
    # https://github.com/numpy/numpy/blob/v1.25.0/numpy/core/function_base.py#L24-L182

    def linspace(self, start, stop, num=50, endpoint=True, retstep=False, dtype=None,
                axis=0):
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

        y = self.arange(0, num, dtype=dt).reshape((-1,) + (1,) * delta.ndim)

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

        y += start

        # FIXME reenable, without in-place ops
        # if endpoint and num > 1:
        #     y[-1, ...] = stop

        if axis != 0:
            # y = _nx.moveaxis(y, 0, axis)
            raise NotImplementedError("axis != 0")

        if integer_dtype:
            y = self.floor(y)  # pylint: disable=no-member

        # FIXME: Use astype
        # https://github.com/inducer/pytato/issues/456
        if retstep:
            return y, step
            # return y.astype(dtype), step
        else:
            return y
            # return y.astype(dtype)

    # }}}

    def arange(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

# }}}


# {{{ BaseFakeNumpyLinalgNamespace

def _reduce_norm(actx, arys, ord):
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
    def __init__(self, array_context):
        self._array_context = array_context

    def norm(self, ary, ord=None):
        if np.isscalar(ary):
            return abs(ary)

        actx = self._array_context

        try:
            from meshmode.dof_array import DOFArray, flat_norm
        except ImportError:
            pass
        else:
            if isinstance(ary, DOFArray):
                from warnings import warn
                warn("Taking an actx.np.linalg.norm of a DOFArray is deprecated. "
                        "(DOFArrays use 2D arrays internally, and "
                        "actx.np.linalg.norm should compute matrix norms of those.) "
                        "This will stop working in 2022. "
                        "Use meshmode.dof_array.flat_norm instead.",
                        DeprecationWarning, stacklevel=2)

                return flat_norm(ary, ord=ord)

        try:
            iterable = serialize_container(ary)
        except NotAnArrayContainerError:
            pass
        else:
            return _reduce_norm(actx, [
                self.norm(subary, ord=ord) for _, subary in iterable
                ], ord=ord)

        if ord is None:
            return self.norm(actx.np.ravel(ary, order="A"), 2)

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
