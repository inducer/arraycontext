"""
An array context is an abstraction that helps you dispatch between multiple
implementations of :mod:`numpy`-like :math:`n`-dimensional arrays.
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

from .context import ArrayContext

from .metadata import CommonSubexpressionTag, FirstAxisIsElementsTag

from .container import (
        ArrayContainer,
        is_array_container, is_array_container_type,
        get_container_context, get_container_context_recursively,
        serialize_container, deserialize_container)
from .container.arithmetic import with_container_arithmetic
from .container.dataclass import dataclass_array_container

from .container.traversal import (
        map_array_container,
        multimap_array_container,
        rec_map_array_container,
        rec_multimap_array_container,
        mapped_over_array_containers,
        multimapped_over_array_containers,
        thaw, freeze,
        from_numpy, to_numpy)

from .impl.pyopencl import PyOpenCLArrayContext

from .pytest import pytest_generate_tests_for_pyopencl_array_context

from .loopy import make_loopy_program


__all__ = (
        "ArrayContext",

        "CommonSubexpressionTag",
        "FirstAxisIsElementsTag",

        "ArrayContainer",
        "is_array_container", "is_array_container_type",
        "get_container_context", "get_container_context_recursively",
        "serialize_container", "deserialize_container",
        "with_container_arithmetic",
        "dataclass_array_container",

        "map_array_container", "multimap_array_container",
        "rec_map_array_container", "rec_multimap_array_container",
        "mapped_over_array_containers",
        "multimapped_over_array_containers",
        "thaw", "freeze",
        "from_numpy", "to_numpy",

        "PyOpenCLArrayContext",

        "make_loopy_program",

        "pytest_generate_tests_for_pyopencl_array_context"
        )


def _acf():
    """A tiny undocumented function to pass to tests that take an ``actx_factory``
    argument when running them from the command line.
    """
    import pyopencl as cl

    context = cl._csc()
    queue = cl.CommandQueue(context)
    return PyOpenCLArrayContext(queue)

# vim: foldmethod=marker
