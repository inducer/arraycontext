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


import numpy as np
from arraycontext.container import is_array_container, serialize_container
from arraycontext.container.traversal import rec_map_array_container


# {{{ BaseFakeNumpyNamespace
class BaseFakeNumpyNamespace:
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
        "sign", "heaviside", "maximum", "fmax", "nan_to_num",

        # FIXME:
        # "interp",
        })
    
    def _new_like(self, ary, alloc_like):
        from numbers import Number

        if isinstance(ary, np.ndarray) and ary.dtype.char == "O":
            # NOTE: we don't want to match numpy semantics on object arrays,
            # e.g. `np.zeros_like(x)` returns `array([0, 0, ...], dtype=object)`
            # FIXME: what about object arrays nested in an ArrayContainer?
            raise NotImplementedError("operation not implemented for object arrays")
        elif is_array_container_type(ary.__class__):
            return rec_map_array_container(alloc_like, ary)
        elif isinstance(ary, Number):
            # NOTE: `np.zeros_like(x)` returns `array(x, shape=())`, which
            # is best implemented by concrete array contexts, if at all
            raise NotImplementedError("operation not implemented for scalars")
        else:
            return alloc_like(ary)

    def empty_like(self, ary):
        return self._new_like(ary, self._array_context.empty_like)

    def zeros_like(self, ary):
        return self._new_like(ary, self._array_context.zeros_like)

    def conjugate(self, x):
        # NOTE: conjugate distributes over object arrays, but it looks for a
        # `conjugate` ufunc, while some implementations only have the shorter
        # `conj` (e.g. cl.array.Array), so this should work for everybody.
        return rec_map_array_container(lambda obj: obj.conj(), x)

    conj = conjugate

# }}}


# {{{ BaseFakeNumpyLinalgNamespace

def _reduce_norm(actx, arys, ord):
    from numbers import Number
    from functools import reduce

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
        from numbers import Number

        if isinstance(ary, Number):
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

        if is_array_container_type(ary.__class__):
            return _reduce_norm(actx, [
                self.norm(subary, ord=ord)
                for _, subary in serialize_container(ary)
                ], ord=ord)

        if ord is None:
            return self.norm(actx.np.ravel(ary, order="A"), 2)

        if len(ary.shape) != 1:
            raise NotImplementedError("only vector norms are implemented")

        if ary.size == 0:
            return ary.dtype.type(0)

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
