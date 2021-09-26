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

from .container import (
    ArrayContainer,
    ArrayContainerT,
    NotAnArrayContainerError,
    deserialize_container,
    get_container_context_opt,
    get_container_context_recursively,
    get_container_context_recursively_opt,
    is_array_container,
    is_array_container_type,
    register_multivector_as_array_container,
    serialize_container,
)
from .container.arithmetic import with_container_arithmetic
from .container.dataclass import dataclass_array_container
from .container.traversal import (
    flat_size_and_dtype,
    flatten,
    freeze,
    from_numpy,
    map_array_container,
    map_reduce_array_container,
    mapped_over_array_containers,
    multimap_array_container,
    multimap_reduce_array_container,
    multimapped_over_array_containers,
    outer,
    rec_map_array_container,
    rec_map_reduce_array_container,
    rec_multimap_array_container,
    rec_multimap_reduce_array_container,
    stringify_array_container_tree,
    thaw,
    to_numpy,
    unflatten,
    with_array_context,
)
from .context import (
    Array,
    ArrayContext,
    ArrayOrContainer,
    ArrayOrContainerOrScalar,
    ArrayOrContainerOrScalarT,
    ArrayOrContainerT,
    ArrayT,
    Scalar,
    ScalarLike,
    tag_axes,
)
from .impl.jax import EagerJAXArrayContext
from .impl.numpy import NumpyArrayContext
from .impl.pyopencl import PyOpenCLArrayContext
from .impl.pytato import PytatoJAXArrayContext, PytatoPyOpenCLArrayContext
from .loopy import make_loopy_program
from .pytest import (
    PytestArrayContextFactory,
    PytestPyOpenCLArrayContextFactory,
    pytest_generate_tests_for_array_contexts,
    pytest_generate_tests_for_pyopencl_array_context,
)
from .transform_metadata import CommonSubexpressionTag, ElementwiseMapKernelTag


__all__ = (
    "Array",
    "ArrayContainer",
    "ArrayContainerT",
    "ArrayContext",
    "ArrayOrContainer",
    "ArrayOrContainerOrScalar",
    "ArrayOrContainerOrScalarT",
    "ArrayOrContainerT",
    "ArrayT",
    "CommonSubexpressionTag",
    "EagerJAXArrayContext",
    "ElementwiseMapKernelTag",
    "NotAnArrayContainerError",
    "NumpyArrayContext",
    "PyOpenCLArrayContext",
    "PytatoJAXArrayContext",
    "PytatoPyOpenCLArrayContext",
    "PytestArrayContextFactory",
    "PytestPyOpenCLArrayContextFactory",
    "Scalar",
    "ScalarLike",
    "dataclass_array_container",
    "deserialize_container",
    "flat_size_and_dtype",
    "flatten",
    "freeze",
    "from_numpy",
    "get_container_context_opt",
    "get_container_context_recursively",
    "get_container_context_recursively_opt",
    "is_array_container",
    "is_array_container_type",
    "make_loopy_program",
    "map_array_container",
    "map_reduce_array_container",
    "mapped_over_array_containers",
    "multimap_array_container",
    "multimap_reduce_array_container",
    "multimapped_over_array_containers",
    "outer",
    "pytest_generate_tests_for_array_contexts",
    "pytest_generate_tests_for_pyopencl_array_context",
    "rec_map_array_container",
    "rec_map_reduce_array_container",
    "rec_multimap_array_container",
    "rec_multimap_reduce_array_container",
    "register_multivector_as_array_container",
    "serialize_container",
    "stringify_array_container_tree",
    "tag_axes",
    "thaw",
    "to_numpy",
    "unflatten",
    "with_array_context",
    "with_container_arithmetic",
    "with_container_arithmetic"
    )


# {{{ deprecation handling

def _deprecated_acf():
    """A tiny undocumented function to pass to tests that take an ``actx_factory``
    argument when running them from the command line.
    """
    import pyopencl as cl

    context = cl._csc()
    queue = cl.CommandQueue(context)
    return PyOpenCLArrayContext(queue)


_depr_name_to_replacement_and_obj = {
        "get_container_context": (
            "get_container_context_opt",
            get_container_context_opt, 2022),
        }


def __getattr__(name):
    replacement_and_obj = _depr_name_to_replacement_and_obj.get(name, None)
    if replacement_and_obj is not None:
        replacement, obj, year = replacement_and_obj
        from warnings import warn
        warn(f"'arraycontext.{name}' is deprecated. "
                f"Use '{replacement}' instead. "
                f"'arraycontext.{name}' will continue to work until {year}.",
                DeprecationWarning, stacklevel=2)
        return obj
    else:
        raise AttributeError(name)

# }}}

# vim: foldmethod=marker
