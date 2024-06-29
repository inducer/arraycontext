"""
.. currentmodule:: arraycontext


A mod :`cupy`-based array context.

.. autoclass:: CupyArrayContext
"""
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

from collections.abc import Mapping


try:
    import cupy as cp  # type: ignore[import-untyped]
except ModuleNotFoundError:
    pass

import loopy as lp

from arraycontext.container.traversal import (
    rec_map_array_container, with_array_context)
from arraycontext.context import ArrayContext


class CupyArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :mod:`cupy.ndarray` to represent arrays


    .. automethod:: __init__
    """
    def __init__(self):
        super().__init__()
        self._loopy_transform_cache: \
                Mapping["lp.TranslationUnit", "lp.TranslationUnit"] = {}

        self.array_types = (cp.ndarray,)

    def _get_fake_numpy_namespace(self):
        from .fake_numpy import CupyFakeNumpyNamespace
        return CupyFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    def clone(self):
        return type(self)()

    def empty(self, shape, dtype):
        return cp.empty(shape, dtype=dtype)

    def zeros(self, shape, dtype):
        return cp.zeros(shape, dtype)

    def from_numpy(self, np_array):
        return cp.array(np_array)

    def to_numpy(self, array):
        return cp.asnumpy(array)

    def call_loopy(self, t_unit, **kwargs):
        t_unit = t_unit.copy(target=lp.ExecutableCTarget())
        try:
            t_unit = self._loopy_transform_cache[t_unit]
        except KeyError:
            orig_t_unit = t_unit
            t_unit = self.transform_loopy_program(t_unit)
            self._loopy_transform_cache[orig_t_unit] = t_unit
            del orig_t_unit

        _, result = t_unit(**kwargs)

        return result

    def freeze(self, array):
        def _freeze(ary):
            return cp.asnumpy(ary)

        return with_array_context(rec_map_array_container(_freeze, array), actx=None)

    def thaw(self, array):
        def _thaw(ary):
            return cp.array(ary)

        return with_array_context(rec_map_array_container(_thaw, array), actx=self)

    # }}}

    def transform_loopy_program(self, t_unit):
        raise ValueError("CupyArrayContext does not implement "
                         "transform_loopy_program. Sub-classes are supposed "
                         "to implement it.")

    def tag(self, tags, array):
        # No tagging support in CupyArrayContext
        return array

    def tag_axis(self, iaxis, tags, array):
        return array

    def einsum(self, spec, *args, arg_names=None, tagged=()):
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
