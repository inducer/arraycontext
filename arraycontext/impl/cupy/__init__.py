"""
.. currentmodule:: arraycontext

A :mod:`cupy`-based array context.

.. autoclass:: CupyArrayContext
"""

from __future__ import annotations


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

from typing import Any, overload

import numpy as np

import loopy as lp
from pytools.tag import ToTagSetConvertible

from arraycontext.container.traversal import rec_map_array_container, with_array_context
from arraycontext.context import (
    Array,
    ArrayContext,
    ArrayOrContainerOrScalar,
    ArrayOrContainerOrScalarT,
    ContainerOrScalarT,
    NumpyOrContainerOrScalar,
)


class CupyNonObjectArrayMetaclass(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        import cupy as cp  # type: ignore[import-untyped]
        return isinstance(instance, cp.ndarray) and instance.dtype != object


class CupyNonObjectArray(metaclass=CupyNonObjectArrayMetaclass):
    pass


class CupyArrayContext(ArrayContext):
    """An :class:`ArrayContext` that uses :class:`cupy.ndarray` to represent arrays."""

    array_types = (CupyNonObjectArray,)

    def _get_fake_numpy_namespace(self):
        from .fake_numpy import CupyFakeNumpyNamespace
        return CupyFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    def clone(self):
        return type(self)()

    @overload
    def from_numpy(self, array: np.ndarray[Any, Any]) -> Array:
        ...

    @overload
    def from_numpy(self, array: ContainerOrScalarT) -> ContainerOrScalarT:
        ...

    def from_numpy(self,
                   array: NumpyOrContainerOrScalar
                   ) -> ArrayOrContainerOrScalar:
        import cupy as cp
        return with_array_context(rec_map_array_container(cp.array, array),
                                  actx=self)

    @overload
    def to_numpy(self, array: Array) -> np.ndarray[Any, Any]:
        ...

    @overload
    def to_numpy(self, array: ContainerOrScalarT) -> ContainerOrScalarT:
        ...

    def to_numpy(self,
                 array: ArrayOrContainerOrScalar
                 ) -> NumpyOrContainerOrScalar:
        import cupy as cp
        return with_array_context(rec_map_array_container(cp.asnumpy, array),
                                  actx=None)

    def call_loopy(
                self,
                t_unit: lp.TranslationUnit, **kwargs: Any
            ) -> dict[str, Array]:
        raise NotImplementedError(
            "Calling loopy on CuPy arrays is not supported. Maybe rewrite"
            " the loopy kernel as numpy-flavored array operations using"
            " ArrayContext.np.")

    def freeze(self, array):
        import cupy as cp
        # Note that we could use a non-blocking version of cp.asnumpy here, but
        # it appears to have very little impact on performance.
        return with_array_context(rec_map_array_container(cp.asnumpy, array), actx=None)

    def thaw(self, array):
        import cupy as cp
        return with_array_context(rec_map_array_container(cp.array, array), actx=self)

    # }}}

    def tag(self,
            tags: ToTagSetConvertible,
            array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        # Cupy (like numpy) doesn't support tagging
        return array

    def tag_axis(self,
                 iaxis: int, tags: ToTagSetConvertible,
                 array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        # Cupy (like numpy) doesn't support tagging
        return array

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        import cupy as cp
        return cp.einsum(spec, *args)

    @property
    def permits_inplace_modification(self):
        return True

    @property
    def supports_nonscalar_broadcasting(self):
        return True

    @property
    def permits_advanced_indexing(self):
        return True
