"""
.. currentmodule:: arraycontext

A :mod:`pytato`-based array context defers the evaluation of an array until its
frozen. The execution contexts for the evaluations are specific to an
:class:`~arraycontext.ArrayContext` type. For ex.
:class:`~arraycontext.PytatoPyOpenCLArrayContext` uses :mod:`pyopencl` to
JIT-compile and execute the array expressions.

Following :mod:`pytato`-based array context are provided:

.. autoclass:: PytatoPyOpenCLArrayContext
.. autoclass:: PytatoJAXArrayContext


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

from arraycontext.context import ArrayContext, _ScalarLike
from arraycontext.container.traversal import rec_map_array_container
from arraycontext.metadata import NameHint

import numpy as np
from typing import Any, Callable, Union, TYPE_CHECKING, Tuple, Type
from pytools.tag import ToTagSetConvertible
import abc

if TYPE_CHECKING:
    import pytato


# {{{ _BasePytatoArrayContext

class _BasePytatoArrayContext(ArrayContext, abc.ABC):
    """
    An abstract :class:`ArrayContext` that uses :mod:`pytato` data types to
    represent.

    .. automethod:: __init__

    .. automethod:: transform_dag

    .. automethod:: compile
    """
    def __init__(self):
        super().__init__()
        self._freeze_prg_cache = {}
        self._dag_transform_cache = {}

    def _get_fake_numpy_namespace(self):
        from arraycontext.impl.pytato.fake_numpy import PytatoFakeNumpyNamespace
        return PytatoFakeNumpyNamespace(self)

    def empty(self, shape, dtype):
        raise ValueError(f"{type(self).__name__} does not support empty")

    def zeros(self, shape, dtype):
        import pytato as pt
        return pt.zeros(shape, dtype)

    def transform_dag(self, dag: "pytato.DictOfNamedArrays"
                      ) -> "pytato.DictOfNamedArrays":
        """
        Returns a transformed version of *dag*. Sub-classes are supposed to
        override this method to implement context-specific transformations on
        *dag* (most likely to perform domain-specific optimizations). Every
        :mod:`pytato` DAG that is compiled to a GPU-kernel is
        passed through this routine.

        :arg dag: An instance of :class:`pytato.DictOfNamedArrays`
        :returns: A transformed version of *dag*.
        """
        return dag

    def transform_loopy_program(self, t_unit):
        raise ValueError(f"{type(self)} does not implement "
                         "transform_loopy_program. Sub-classes are supposed "
                         "to implement it.")

    @abc.abstractproperty
    def frozen_array_types(self) -> Tuple[Type, ...]:
        """
        Returns valid frozen array types for the array context.
        """
        pass

    @abc.abstractmethod
    def einsum(self, spec, *args, arg_names=None, tagged=()):
        pass

    @property
    def permits_inplace_modification(self):
        return False

    @property
    def supports_nonscalar_broadcasting(self):
        return True

    @property
    def permits_advanced_indexing(self):
        return True

# }}}


# {{{ PytatoPyOpenCLArrayContext

class PytatoPyOpenCLArrayContext(_BasePytatoArrayContext):
    """
    A :class:`ArrayContext` that uses :mod:`pytato` data types to represent
    the arrays targeting OpenCL for offloading operations.

    .. attribute:: queue

        A :class:`pyopencl.CommandQueue`.

    .. attribute:: allocator

        A :mod:`pyopencl` memory allocator. Can also be None (default) or False
        to use the default allocator.

    .. automethod:: __init__

    .. automethod:: transform_dag

    .. automethod:: compile
    """

    def __init__(self, queue, allocator=None):
        import pytato as pt
        import pyopencl.array as cla
        super().__init__()
        self.queue = queue
        self.allocator = allocator
        self.array_types = (pt.Array, cla.Array)

        # unused, but necessary to keep the context alive
        self.context = self.queue.context

    # {{{ ArrayContext interface

    def clone(self):
        return type(self)(self.queue, self.allocator)

    def from_numpy(self, array: Union[np.ndarray, _ScalarLike]):
        import pytato as pt
        import pyopencl.array as cla
        cl_array = cla.to_device(self.queue, array)
        return pt.make_data_wrapper(cl_array)

    def to_numpy(self, array):
        if np.isscalar(array):
            return array

        cl_array = self.freeze(array)
        return cl_array.get(queue=self.queue)

    @property
    def frozen_array_types(self) -> Tuple[Type, ...]:
        import pyopencl.array as cla
        return (cla.Array, )

    def call_loopy(self, program, **kwargs):
        import pytato as pt
        from pytato.scalar_expr import SCALAR_CLASSES
        from pytato.loopy import call_loopy
        from arraycontext.impl.pyopencl.taggable_cl_array import TaggableCLArray

        entrypoint = program.default_entrypoint.name

        # {{{ preprocess args

        processed_kwargs = {}

        for kw, arg in sorted(kwargs.items()):
            if isinstance(arg, (pt.Array,) + SCALAR_CLASSES):
                pass
            elif isinstance(arg, TaggableCLArray):
                arg = self.thaw(arg)
            else:
                raise ValueError(f"call_loopy argument '{kw}' expected to be an"
                                 " instance of 'pytato.Array', 'Number' or"
                                 f"'TaggableCLArray', got '{type(arg)}'")

            processed_kwargs[kw] = arg

        # }}}

        return call_loopy(program, processed_kwargs, entrypoint)

    def freeze(self, array):
        import pytato as pt
        import pyopencl.array as cla
        import loopy as lp
        from arraycontext.impl.pytato.utils import (_normalize_pt_expr,
                                                    get_cl_axes_from_pt_axes)
        from arraycontext.impl.pyopencl.taggable_cl_array import (to_tagged_cl_array,
                                                                  TaggableCLArray)

        if isinstance(array, TaggableCLArray):
            return array.with_queue(None)
        if isinstance(array, cla.Array):
            from warnings import warn
            warn("Freezing pyopencl.array.Array will be deprecated in 2023."
                 " Use `to_tagged_cl_array` to convert the array to"
                 " TaggableCLArray", DeprecationWarning, stacklevel=2)
            return to_tagged_cl_array(array.with_queue(None),
                                      axes=None,
                                      tags=frozenset())
        if isinstance(array, pt.DataWrapper):
            # trivial freeze.
            return to_tagged_cl_array(array.data.with_queue(None),
                                      axes=get_cl_axes_from_pt_axes(array.axes),
                                      tags=array.tags)
        if not isinstance(array, pt.Array):
            raise TypeError(f"{type(self).__name__}.freeze invoked "
                            f"with non-pytato array of type '{type(array)}'")

        # {{{ early exit for 0-sized arrays

        if array.size == 0:
            return to_tagged_cl_array(
                cla.empty(self.queue.context,
                          shape=array.shape,
                          dtype=array.dtype,
                          allocator=self.allocator),
                get_cl_axes_from_pt_axes(array.axes),
                array.tags)

        # }}}

        pt_dict_of_named_arrays = pt.make_dict_of_named_arrays(
                {"_actx_out": array})

        normalized_expr, bound_arguments = _normalize_pt_expr(
                pt_dict_of_named_arrays)

        try:
            pt_prg = self._freeze_prg_cache[normalized_expr]
        except KeyError:
            if normalized_expr in self._dag_transform_cache:
                transformed_dag = self._dag_transform_cache[normalized_expr]
            else:
                transformed_dag = self.transform_dag(normalized_expr)
                self._dag_transform_cache[normalized_expr] = transformed_dag

            pt_prg = pt.generate_loopy(transformed_dag,
                                       options=lp.Options(return_dict=True,
                                                          no_numpy=True),
                                       cl_device=self.queue.device)
            pt_prg = pt_prg.with_transformed_program(self.transform_loopy_program)
            self._freeze_prg_cache[normalized_expr] = pt_prg

        assert len(pt_prg.bound_arguments) == 0
        evt, out_dict = pt_prg(self.queue, **bound_arguments)
        evt.wait()

        return to_tagged_cl_array(
            out_dict["_actx_out"].with_queue(None),
            get_cl_axes_from_pt_axes(
                self._dag_transform_cache[normalized_expr]["_actx_out"].expr.axes),
            array.tags)

    def thaw(self, array):
        import pytato as pt
        from .utils import get_pt_axes_from_cl_axes
        from arraycontext.impl.pyopencl.taggable_cl_array import (TaggableCLArray,
                                                                  to_tagged_cl_array)
        import pyopencl.array as cl_array

        if isinstance(array, TaggableCLArray):
            pass
        elif isinstance(array, cl_array.Array):
            array = to_tagged_cl_array(array, axes=None, tags=frozenset())
        else:
            raise TypeError(f"{type(self).__name__}.thaw expects "
                            "'TaggableCLArray' or 'cl.array.Array' got "
                            f"{type(array)}.")

        return pt.make_data_wrapper(array.with_queue(self.queue),
                                    axes=get_pt_axes_from_cl_axes(array.axes),
                                    tags=array.tags)

    # }}}

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        from .compile import LazilyPyOpenCLCompilingFunctionCaller
        return LazilyPyOpenCLCompilingFunctionCaller(self, f)

    def transform_dag(self, dag: "pytato.DictOfNamedArrays"
                      ) -> "pytato.DictOfNamedArrays":
        import pytato as pt
        dag = pt.transform.materialize_with_mpms(dag)
        return dag

    def tag(self, tags: ToTagSetConvertible, array):
        return rec_map_array_container(lambda x: x.tagged(tags),
                                       array)

    def tag_axis(self, iaxis, tags: ToTagSetConvertible, array):
        return rec_map_array_container(
            lambda x: x.with_tagged_axis(iaxis, tags),
            array)

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        import pyopencl.array as cla
        import pytato as pt
        from arraycontext.impl.pyopencl.taggable_cl_array import (TaggableCLArray,
                                                                  to_tagged_cl_array)
        if arg_names is None:
            arg_names = (None,) * len(args)

        def preprocess_arg(name, arg):
            if isinstance(arg, TaggableCLArray):
                ary = self.thaw(arg)
            elif isinstance(arg, cla.Array):
                from warnings import warn
                warn("Passing pyopencl.array.Array to einsum will be "
                     "deprecated in 2023."
                     " Use `to_tagged_cl_array` to convert the array to"
                     " TaggableCLArray.", DeprecationWarning, stacklevel=2)
                ary = self.thaw(to_tagged_cl_array(arg,
                                                   axes=None,
                                                   tags=frozenset()))
            else:
                assert isinstance(arg, pt.Array)
                ary = arg

            if name is not None:
                # Tagging Placeholders with naming-related tags is pointless:
                # They already have names. It's also counterproductive, as
                # multiple placeholders with the same name that are not
                # also the same object are not allowed, and this would produce
                # a different Placeholder object of the same name.
                if (not isinstance(ary, pt.Placeholder)
                        and not ary.tags_of_type(NameHint)):
                    ary = ary.tagged(NameHint(name))

            return ary

        return pt.einsum(spec, *[
            preprocess_arg(name, arg)
            for name, arg in zip(arg_names, args)
            ])

# }}}


# {{{ PytatoJAXArrayContext

class PytatoJAXArrayContext(_BasePytatoArrayContext):
    """
    An arraycontext that uses :mod:`pytato` to represent the thawed state of
    the arrays and compiles the expressions using
    :class:`pytato.target.python.JAXPythonTarget`.
    """

    def __init__(self):
        import pytato as pt
        from jax.numpy import DeviceArray
        super().__init__()
        self.array_types = (pt.Array, DeviceArray)

    def clone(self):
        return type(self)()

    def from_numpy(self, array: Union[np.ndarray, _ScalarLike]):
        import jax
        import pytato as pt
        return pt.make_data_wrapper(jax.device_put(array))

    def to_numpy(self, array):
        if np.isscalar(array):
            return array

        import jax
        return jax.device_get(self.freeze(array))

    @property
    def frozen_array_types(self) -> Tuple[Type, ...]:
        from jax.numpy import DeviceArray
        return (DeviceArray, )

    def call_loopy(self, program, **kwargs):
        raise ValueError(f"{type(self)} does not support calling loopy.")

    def freeze(self, array):
        import pytato as pt
        from jax.numpy import DeviceArray

        if isinstance(array, DeviceArray):
            return array.block_until_ready()
        if not isinstance(array, pt.Array):
            raise TypeError(f"{type(self)}.freeze invoked with "
                            f"non-pytato array of type '{type(array)}'")

        from arraycontext.impl.pytato.utils import _normalize_pt_expr
        pt_dict_of_named_arrays = pt.make_dict_of_named_arrays(
                {"_actx_out": array})

        normalized_expr, bound_arguments = _normalize_pt_expr(
                pt_dict_of_named_arrays)

        try:
            pt_prg = self._freeze_prg_cache[normalized_expr]
        except KeyError:
            pt_prg = pt.generate_jax(self.transform_dag(normalized_expr),
                                     jit=True)
            self._freeze_prg_cache[normalized_expr] = pt_prg

        assert len(pt_prg.bound_arguments) == 0
        out_dict = pt_prg(**bound_arguments)

        return out_dict["_actx_out"].block_until_ready()

    def thaw(self, array):
        import pytato as pt

        if not isinstance(array, self.frozen_array_types):
            raise TypeError(f"{type(self)}.thaw expects jax device arrays, got "
                            f"{type(array)}")

        return pt.make_data_wrapper(array)

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        from .compile import LazilyJAXCompilingFunctionCaller
        return LazilyJAXCompilingFunctionCaller(self, f)

    def tag(self, tags: ToTagSetConvertible, array):
        import pytato as pt
        from jax.numpy import DeviceArray

        def _rec_tag(ary):
            if isinstance(ary, DeviceArray):
                return ary
            else:
                assert isinstance(ary, pt.Array)
                return ary.tagged(tags)

        return rec_map_array_container(_rec_tag, array)

    def tag_axis(self, iaxis, tags: ToTagSetConvertible, array):
        import pytato as pt
        from jax.numpy import DeviceArray

        def _rec_tag_axis(ary):
            if isinstance(ary, DeviceArray):
                return ary
            else:
                assert isinstance(ary, pt.Array)
                return ary.with_tagged_axis(iaxis, tags)

        return rec_map_array_container(_rec_tag_axis,
                                       array)

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        import pytato as pt
        from jax.numpy import DeviceArray
        if arg_names is None:
            arg_names = (None,) * len(args)

        def preprocess_arg(name, arg):
            if isinstance(arg, DeviceArray):
                ary = self.thaw(arg)
            else:
                assert isinstance(arg, pt.Array)
                ary = arg

            if name is not None:
                # Tagging Placeholders with naming-related tags is pointless:
                # They already have names. It's also counterproductive, as
                # multiple placeholders with the same name that are not
                # also the same object are not allowed, and this would produce
                # a different Placeholder object of the same name.
                if (not isinstance(ary, pt.Placeholder)
                        and not ary.tags_of_type(NameHint)):
                    ary = ary.tagged(NameHint(name))

            return ary

        return pt.einsum(spec, *[
            preprocess_arg(name, arg)
            for name, arg in zip(arg_names, args)
            ])

# }}}


# vim: foldmethod=marker
