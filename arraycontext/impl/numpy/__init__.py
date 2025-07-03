"""
.. currentmodule:: arraycontext

A :mod:`numpy`-based array context.

.. autoclass:: NumpyArrayContext
"""

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

from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
from typing_extensions import override

import loopy as lp

from arraycontext.container.traversal import (
    rec_map_array_container as rec_map_array_container,
    rec_map_container,
    with_array_context,
)
from arraycontext.context import (
    ArrayContext,
    UntransformedCodeWarning,
)
from arraycontext.typing import (
    Array,
    ArrayOrContainerOrScalar,
    ArrayOrContainerOrScalarT,
    ContainerOrScalarT,
    NumpyOrContainerOrScalar,
    is_scalar_like,
)


if TYPE_CHECKING:
    from pymbolic import Scalar
    from pytools.tag import ToTagSetConvertible

    from arraycontext.typing import ArrayContainerT


class NumpyNonObjectArrayMetaclass(type):
    @override
    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, np.ndarray) and instance.dtype != object


class NumpyNonObjectArray(metaclass=NumpyNonObjectArrayMetaclass):
    pass


class NumpyArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :class:`numpy.ndarray` to represent arrays.

    .. automethod:: __init__
    """

    _loopy_transform_cache: dict[lp.TranslationUnit, lp.ExecutorBase]

    def __init__(self) -> None:
        super().__init__()
        self._loopy_transform_cache = {}

    array_types = (NumpyNonObjectArray,)

    def _get_fake_numpy_namespace(self):
        from .fake_numpy import NumpyFakeNumpyNamespace
        return NumpyFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    @override
    def clone(self):
        return type(self)()

    @overload
    def from_numpy(self, array: np.ndarray) -> Array:
        ...

    @overload
    def from_numpy(self, array: Scalar) -> Array:
        ...

    @overload
    def from_numpy(self, array: ArrayContainerT) -> ArrayContainerT:
        ...

    @override
    def from_numpy(self,
                   array: NumpyOrContainerOrScalar
                   ) -> ArrayOrContainerOrScalar:
        if isinstance(array, np.ndarray) or is_scalar_like(array):
            return cast("Array", cast("object", np.array(array)))
        return array

    @overload
    def to_numpy(self, array: Array) -> np.ndarray:
        ...

    @overload
    def to_numpy(self, array: ContainerOrScalarT) -> ContainerOrScalarT:
        ...

    @override
    def to_numpy(self,
                 array: ArrayOrContainerOrScalar
                 ) -> NumpyOrContainerOrScalar:
        return cast("NumpyOrContainerOrScalar", array)

    @override
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

    @override
    def freeze(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        def _freeze(ary):
            return ary

        return with_array_context(rec_map_container(_freeze, array), actx=None)

    @override
    def thaw(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        def _thaw(ary):
            return ary

        return with_array_context(rec_map_container(_thaw, array), actx=self)

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
        # Numpy doesn't support tagging
        return array

    def tag_axis(self,
                 iaxis: int, tags: ToTagSetConvertible,
                 array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        # Numpy doesn't support tagging
        return array

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        return np.einsum(spec, *args)

    @property
    def permits_inplace_modification(self):
        return True

    @property
    def supports_nonscalar_broadcasting(self):
        return True

    @property
    def permits_advanced_indexing(self):
        return True
