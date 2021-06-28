"""
.. currentmodule:: arraycontext

A :mod:`pytato`-based array context defers the evaluation of an array until its
frozen. The execution contexts for the evaluations are specific to an
:class:`~arraycontext.ArrayContext` type. For ex.
:class:`~arraycontext.PytatoPyOpenCLArrayContext` uses :mod:`pyopencl` to
JIT-compile and execute the array expressions.

Following :mod:`pytato`-based array context are provided:

.. autoclass:: PytatoPyOpenCLArrayContext


Compiling a python callable
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: arraycontext.impl.pytato.compile
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

from arraycontext.context import ArrayContext
import numpy as np
from typing import Any, Callable, Union, Sequence
from pytools.tag import Tag


class PytatoPyOpenCLArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :mod:`pytato` data types to represent
    the arrays targeting OpenCL for offloading operations.

    .. attribute:: queue

        A :class:`pyopencl.CommandQueue`.

    .. attribute:: allocator

        A :mod:`pyopencl` memory allocator. Can also be None (default) or False
        to use the default allocator.

    .. automethod:: __init__
    """

    def __init__(self, queue, allocator=None):
        super().__init__()
        self.queue = queue
        self.allocator = allocator

        # unused, but necessary to keep the context alive
        self.context = self.queue.context

    def _get_fake_numpy_namespace(self):
        from arraycontext.impl.pytato.fake_numpy import PytatoFakeNumpyNamespace
        return PytatoFakeNumpyNamespace(self)

    def get_array_types(self):
        """
        Returns a :class:`tuple` of types that are valid base array classes
        the context can operate on.
        """
        import pytato as pt
        return (pt.Array,)

    # {{{ ArrayContext interface

    def clone(self):
        return type(self)(self.queue, self.allocator)

    def empty(self, shape, dtype):
        raise ValueError("PytatoPyOpenCLArrayContext does not support empty")

    def zeros(self, shape, dtype):
        import pytato as pt
        return pt.zeros(shape, dtype)

    def from_numpy(self, np_array: np.ndarray):
        import pytato as pt
        import pyopencl.array as cla
        cl_array = cla.to_device(self.queue, np_array)
        return pt.make_data_wrapper(cl_array)

    def to_numpy(self, array):
        cl_array = self.freeze(array)
        return cl_array.get(queue=self.queue)

    def call_loopy(self, program, **kwargs):
        import pyopencl.array as cla
        from pytato.loopy import call_loopy
        entrypoint, = set(program.callables_table)

        # thaw frozen arrays
        kwargs = {kw: (self.thaw(arg) if isinstance(arg, cla.Array) else arg)
                  for kw, arg in kwargs.items()}

        return call_loopy(program, kwargs, entrypoint)

    def freeze(self, array):
        # TODO: This should store a cache of pytato DAG -> build pyopencl
        # program instead of re-compiling the DAG for every freeze.

        import pytato as pt
        import pyopencl.array as cla

        if isinstance(array, cla.Array):
            return array.with_queue(None)
        if not isinstance(array, pt.Array):
            raise TypeError("PytatoPyOpenCLArrayContext.freeze invoked with "
                            f"non-pytato array of type '{type(array)}'")

        pt_prg = pt.generate_loopy(array, cl_device=self.queue.device)
        pt_prg = pt_prg.with_transformed_program(self.transform_loopy_program)

        evt, (cl_array,) = pt_prg(self.queue)
        evt.wait()

        return cl_array.with_queue(None)

    def thaw(self, array):
        import pytato as pt
        import pyopencl.array as cla

        if not isinstance(array, cla.Array):
            raise TypeError("PytatoPyOpenCLArrayContext.thaw expects CL arrays, got "
                    f"{type(array)}")

        return pt.make_data_wrapper(array.with_queue(self.queue))

    # }}}

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        from arraycontext.impl.pytato.compile import LazilyCompilingFunctionCaller
        return LazilyCompilingFunctionCaller(self, f)

    def transform_loopy_program(self, t_unit):
        raise ValueError("PytatoPyOpenCLArrayContext does not implement "
                         "transform_loopy_program. Sub-classes are supposed "
                         "to implement it.")

    def tag(self, tags: Union[Sequence[Tag], Tag], array):
        return array.tagged(tags)

    def tag_axis(self, iaxis, tags: Union[Sequence[Tag], Tag], array):
        # TODO
        from warnings import warn
        warn("tagging PytatoPyOpenCLArrayContext's array axes: not yet implemented",
             stacklevel=2)
        return array

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        import pyopencl.array as cla
        import pytato as pt
        if arg_names is not None:
            from warnings import warn
            warn("'arg_names' don't bear any significance in "
                 "PytatoPyOpenCLArrayContext.", stacklevel=2)

        def preprocess_arg(arg):
            if isinstance(arg, cla.Array):
                return self.thaw(arg)
            else:
                assert isinstance(arg, pt.Array)
                return arg

        return pt.einsum(spec, *(preprocess_arg(arg) for arg in args))

    @property
    def permits_inplace_modification(self):
        return False
