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
    UntransformedCodeWarning,
)


class CupyNonObjectArrayMetaclass(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        import cupy as cp  # type: ignore[import-untyped]
        return isinstance(instance, cp.ndarray) and instance.dtype != object


class CupyNonObjectArray(metaclass=CupyNonObjectArrayMetaclass):
    pass


class CupyArrayContext(ArrayContext):
    """
    An :class:`ArrayContext` that uses :class:`cupy.ndarray` to represent arrays.

    .. automethod:: __init__
    """

    _loopy_transform_cache: dict[lp.TranslationUnit, lp.ExecutorBase]

    def __init__(self) -> None:
        super().__init__()
        self._loopy_transform_cache = {}

    array_types = (CupyNonObjectArray,)

    def _get_fake_numpy_namespace(self):
        from .fake_numpy import CupyFakeNumpyNamespace
        return CupyFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    def clone(self):
        return type(self)()

    @overload
    def from_numpy(self, array: np.ndarray) -> Array:
        ...

    @overload
    def from_numpy(self, array: ContainerOrScalarT) -> ContainerOrScalarT:
        ...

    def from_numpy(self,
                   array: NumpyOrContainerOrScalar
                   ) -> ArrayOrContainerOrScalar:
        import cupy as cp
        return cp.array(array)

    @overload
    def to_numpy(self, array: Array) -> np.ndarray:
        ...

    @overload
    def to_numpy(self, array: ContainerOrScalarT) -> ContainerOrScalarT:
        ...

    def to_numpy(self,
                 array: ArrayOrContainerOrScalar
                 ) -> NumpyOrContainerOrScalar:
        import cupy as cp
        return cp.asnumpy(array)

    def call_loopy(
                self,
                t_unit: lp.TranslationUnit, **kwargs: Any
            ) -> dict[str, Array]:
        t_unit = t_unit.copy(target=lp.ExecutableCTarget())
        try:
            executor = self._loopy_transform_cache[t_unit]
        except KeyError:
            executor = self.transform_loopy_program(t_unit).executor()
            self._loopy_transform_cache[t_unit] = executor

        _, result = executor(**kwargs)

        return result

    def freeze(self, array):
        import cupy as cp

        def _freeze(ary):
            return cp.asnumpy(ary)

        return with_array_context(rec_map_array_container(_freeze, array), actx=None)

    def thaw(self, array):
        import cupy as cp

        def _thaw(ary):
            return cp.array(ary)

        return with_array_context(rec_map_array_container(_thaw, array), actx=self)

    # }}}

    def transform_loopy_program(self, t_unit):
        from warnings import warn
        warn("Using the base "
                f"{type(self).__name__}.transform_loopy_program "
                "to transform a translation unit. "
                "This is a no-op and will result in unoptimized C code for"
                "the requested optimization, all in a single statement."
                "This will work, but is unlikely to be performant."
                f"Instead, subclass {type(self).__name__} and implement "
                "the specific transform logic required to transform the program "
                "for your package or application. Check higher-level packages "
                "(e.g. meshmode), which may already have subclasses you may want "
                "to build on.",
                UntransformedCodeWarning, stacklevel=2)

        return t_unit

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