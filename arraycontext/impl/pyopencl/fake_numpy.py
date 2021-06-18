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

from functools import partial
import operator

import numpy as np

from arraycontext.fake_numpy import \
        BaseFakeNumpyNamespace, BaseFakeNumpyLinalgNamespace
from arraycontext.container.traversal import (rec_multimap_array_container,
                                              rec_map_array_container)
from arraycontext.container import serialize_container, is_array_container

try:
    import pyopencl as cl  # noqa: F401
    import pyopencl.array as cl_array
except ImportError:
    pass


# {{{ fake numpy

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

    def sum(self, a, dtype=None):
        return cl_array.sum(
                a, dtype=dtype, queue=self._array_context.queue).get()[()]

    def min(self, a):
        return cl_array.min(a, queue=self._array_context.queue).get()[()]

    def max(self, a):
        return cl_array.max(a, queue=self._array_context.queue).get()[()]

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
                lambda *args: cl_array.stack(arrays=args, axis=axis,
                    queue=self._array_context.queue),
                *arrays)

    def reshape(self, a, newshape):
        return cl_array.reshape(a, newshape)

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

# }}}


# {{{ fake np.linalg

def _flatten_array(ary):
    assert isinstance(ary, cl_array.Array)

    if ary.size == 0:
        # Work around https://github.com/inducer/pyopencl/pull/402
        return ary._new_with_changes(
                data=None, offset=0, shape=(0,), strides=(ary.dtype.itemsize,))
    if ary.flags.f_contiguous:
        return ary.reshape(-1, order="F")
    elif ary.flags.c_contiguous:
        return ary.reshape(-1, order="C")
    else:
        raise ValueError("cannot flatten array "
                f"with strides {ary.strides} of {ary.dtype}")


class _PyOpenCLFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    def norm(self, ary, ord=None):
        from numbers import Number

        if isinstance(ary, Number):
            return abs(ary)

        if ord is None and isinstance(ary, cl_array.Array):
            if ary.ndim == 1:
                ord = 2
            else:
                # mimics numpy's norm computation
                return self.norm(_flatten_array(ary), ord=2)

        try:
            from meshmode.dof_array import DOFArray
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

                import numpy.linalg as la
                return la.norm(
                        [self.norm(_flatten_array(subary), ord=ord)
                            for _, subary in serialize_container(ary)],
                        ord=ord)

        if is_array_container(ary):
            import numpy.linalg as la
            return la.norm(
                    [self.norm(subary, ord=ord)
                        for _, subary in serialize_container(ary)],
                    ord=ord)

        if len(ary.shape) != 1:
            raise NotImplementedError("only vector norms are implemented")

        if ary.size == 0:
            return 0

        if ord == np.inf:
            return self._array_context.np.max(abs(ary))
        elif isinstance(ord, Number) and ord > 0:
            return self._array_context.np.sum(abs(ary)**ord)**(1/ord)
        else:
            raise NotImplementedError(f"unsupported value of 'ord': {ord}")

# }}}


# vim: foldmethod=marker
