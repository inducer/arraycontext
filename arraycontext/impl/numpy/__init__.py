"""
.. currentmodule:: arraycontext

A mod :`numpy`-based array context.

.. autoclass:: NumpyArrayContext
"""

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

from typing import Dict, Sequence, Union

import numpy as np

import loopy as lp
from pytools.tag import Tag

from arraycontext.context import ArrayContext


class NumpyArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :class:`numpy.ndarray` to represent arrays.

    .. automethod:: __init__
    """
    def __init__(self):
        super().__init__()
        self._loopy_transform_cache: \
                Dict[lp.TranslationUnit, lp.TranslationUnit] = {}

        self.array_types = (np.ndarray,)

    def _get_fake_numpy_namespace(self):
        from .fake_numpy import NumpyFakeNumpyNamespace
        return NumpyFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    def clone(self):
        return type(self)()

    def empty(self, shape, dtype):
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype)

    def from_numpy(self, np_array: np.ndarray):
        # Uh oh...
        return np_array

    def to_numpy(self, array):
        # Uh oh...
        return array

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
        return array

    def thaw(self, array):
        return array

    # }}}

    def transform_loopy_program(self, t_unit):
        raise ValueError("NumpyArrayContext does not implement "
                         "transform_loopy_program. Sub-classes are supposed "
                         "to implement it.")

    def tag(self, tags: Union[Sequence[Tag], Tag], array):
        # Numpy doesn't support tagging
        return array

    def tag_axis(self, iaxis, tags: Union[Sequence[Tag], Tag], array):
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
