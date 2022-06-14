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


Compiling a Python callable (Internal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

import sys
from arraycontext.context import ArrayContext, _ScalarLike
from arraycontext.container.traversal import (rec_map_array_container,
                                              with_array_context)
from arraycontext.metadata import NameHint

import numpy as np
from typing import (Any, Callable, Union, TYPE_CHECKING, Tuple, Type, FrozenSet,
        Dict, Optional)
from pytools.tag import ToTagSetConvertible, normalize_tags, Tag
import abc

if TYPE_CHECKING:
    import pytato
    import pyopencl as cl

if getattr(sys, "ARRAYCONTEXT_BUILDING_SPHINX_DOCS", False):
    import pyopencl as cl  # noqa: F811


# {{{ tag conversion

def _preprocess_array_tags(tags: ToTagSetConvertible) -> FrozenSet[Tag]:
    tags = normalize_tags(tags)

    name_hints = [tag for tag in tags if isinstance(tag, NameHint)]
    if name_hints:
        name_hint, = name_hints

        from pytato.tags import PrefixNamed
        prefix_nameds = [tag for tag in tags if isinstance(tag, PrefixNamed)]

        if prefix_nameds:
            prefix_named, = prefix_nameds
            from warnings import warn
            warn("When converting a "
                    f"arraycontext.metadata.NameHint('{name_hint.name}') "
                    "to pytato.tags.PrefixNamed, "
                    f"PrefixNamed('{prefix_named.prefix}') "
                    "was already present.")

        tags = (
                (tags | frozenset({PrefixNamed(name_hint.name)}))
                - {name_hint})

    return tags

# }}}


# {{{ _BasePytatoArrayContext

class _BasePytatoArrayContext(ArrayContext, abc.ABC):
    """
    An abstract :class:`ArrayContext` that uses :mod:`pytato` data types to
    represent.

    .. automethod:: __init__

    .. automethod:: transform_dag

    .. automethod:: compile
    """
    def __init__(self,
            *, compile_trace_callback: Optional[Callable[[Any, str, Any], None]]
             = None) -> None:
        """
        :arg compile_trace_callback: A function of three arguments
            *(what, stage, ir)*, where *what* identifies the object
            being compiled, *stage* is a string describing the compilation
            pass, and *ir* is an object containing the intermediate
            representation. This interface should be considered
            unstable.
        """
        import pytato as pt
        import loopy as lp
        super().__init__()
        self._freeze_prg_cache: Dict[pt.DictOfNamedArrays, lp.TranslationUnit] = {}
        self._dag_transform_cache: Dict[
                pt.DictOfNamedArrays,
                Tuple[pt.DictOfNamedArrays, str]] = {}

        if compile_trace_callback is None:
            def _compile_trace_callback(what, stage, ir):
                pass

            compile_trace_callback = _compile_trace_callback

        self._compile_trace_callback = compile_trace_callback

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
    def __init__(self, queue: "cl.CommandQueue", allocator=None,
            *,
            compile_trace_callback: Optional[Callable[[Any, str, Any], None]]
             = None) -> None:
        """
        :arg compile_trace_callback: A function of three arguments
            *(what, stage, ir)*, where *what* identifies the object
            being compiled, *stage* is a string describing the compilation
            pass, and *ir* is an object containing the intermediate
            representation. This interface should be considered
            unstable.
        """
        import pytato as pt
        import pyopencl.array as cla
        super().__init__(compile_trace_callback=compile_trace_callback)
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

        from arraycontext.container import ArrayT
        from arraycontext.container.traversal import rec_keyed_map_array_container
        from arraycontext.impl.pytato.utils import (_normalize_pt_expr,
                                                    get_cl_axes_from_pt_axes)
        from arraycontext.impl.pyopencl.taggable_cl_array import (to_tagged_cl_array,
                                                                  TaggableCLArray)
        from arraycontext.impl.pytato.compile import _ary_container_key_stringifier

        array_as_dict: Dict[str, Union[cla.Array, TaggableCLArray,
                                       pt.Array]] = {}
        key_to_frozen_subary: Dict[str, TaggableCLArray] = {}
        key_to_pt_arrays: Dict[str, pt.Array] = {}

        def _record_leaf_ary_in_dict(key: Tuple[Any, ...],
                                     ary: ArrayT):
            key_str = "_ary" + _ary_container_key_stringifier(key)
            array_as_dict[key_str] = ary
            return ary

        rec_keyed_map_array_container(_record_leaf_ary_in_dict, array)

        # {{{ remove any non pytato arrays from array_as_dict

        for key, subary in array_as_dict.items():
            if isinstance(subary, TaggableCLArray):
                key_to_frozen_subary[key] = subary.with_queue(None)
            elif isinstance(subary, cla.Array):
                from warnings import warn
                warn("Freezing pyopencl.array.Array will be deprecated in 2023."
                     " Use `to_tagged_cl_array` to convert the array to"
                     " TaggableCLArray", DeprecationWarning, stacklevel=2)
                key_to_frozen_subary[key] = to_tagged_cl_array(
                    subary.with_queue(None),
                    axes=None,
                    tags=frozenset())
            elif isinstance(subary, pt.DataWrapper):
                # trivial freeze.
                key_to_frozen_subary[key] = to_tagged_cl_array(
                    subary.data,
                    axes=get_cl_axes_from_pt_axes(subary.axes),
                    tags=subary.tags)
            else:
                if not isinstance(subary, pt.Array):
                    raise TypeError(f"{type(self).__name__}.freeze invoked "
                                    f"with non-pytato array of type '{type(array)}'")

                # Don't be tempted to take shortcuts here, e.g. for empty
                # arrays, as this will inhibit metadata propagation that
                # may happen in transform_dag below. See
                # https://github.com/inducer/arraycontext/pull/167#issuecomment-1151877480
                key_to_pt_arrays[key] = subary

        # }}}

        pt_dict_of_named_arrays = pt.make_dict_of_named_arrays(
            key_to_pt_arrays)

        normalized_expr, bound_arguments = _normalize_pt_expr(
                pt_dict_of_named_arrays)

        try:
            pt_prg = self._freeze_prg_cache[normalized_expr]
        except KeyError:
            try:
                transformed_dag, function_name = \
                        self._dag_transform_cache[normalized_expr]
            except KeyError:
                transformed_dag = self.transform_dag(normalized_expr)

                from pytato.tags import PrefixNamed
                name_hint_tags = []
                for subary in key_to_pt_arrays.values():
                    name_hint_tags.extend(subary.tags_of_type(PrefixNamed))

                from pytools import common_prefix
                name_hint = common_prefix([nh.prefix for nh in name_hint_tags])
                if name_hint:
                    # All name_hint_tags shared at least some common prefix.
                    function_name = f"frozen_{name_hint}"
                else:
                    function_name = "frozen_result"

                self._dag_transform_cache[normalized_expr] = (
                        transformed_dag, function_name)

            pt_prg = pt.generate_loopy(transformed_dag,
                                       options=lp.Options(return_dict=True,
                                                          no_numpy=True),
                                       cl_device=self.queue.device,
                                       function_name=function_name)
            pt_prg = pt_prg.with_transformed_program(self.transform_loopy_program)
            self._freeze_prg_cache[normalized_expr] = pt_prg
        else:
            transformed_dag, function_name = \
                    self._dag_transform_cache[normalized_expr]

        assert len(pt_prg.bound_arguments) == 0
        evt, out_dict = pt_prg(self.queue, **bound_arguments)
        evt.wait()
        assert len(set(out_dict) & set(key_to_frozen_subary)) == 0

        key_to_frozen_subary = {
            **key_to_frozen_subary,
            **{k: to_tagged_cl_array(v.with_queue(None),
                                     get_cl_axes_from_pt_axes(transformed_dag[k]
                                                              .expr
                                                              .axes),
                                     transformed_dag[k].expr.tags)
               for k, v in out_dict.items()}
        }

        def _to_frozen(key: Tuple[Any, ...], ary: ArrayT):
            key_str = "_ary" + _ary_container_key_stringifier(key)
            return key_to_frozen_subary[key_str]

        return with_array_context(rec_keyed_map_array_container(_to_frozen,
                                                                array),
                                  actx=None)

    def thaw(self, array):
        import pytato as pt
        from .utils import get_pt_axes_from_cl_axes
        from arraycontext.impl.pyopencl.taggable_cl_array import (TaggableCLArray,
                                                                  to_tagged_cl_array)
        import pyopencl.array as cl_array

        def _rec_thaw(ary):
            if isinstance(ary, TaggableCLArray):
                pass
            elif isinstance(ary, cl_array.Array):
                ary = to_tagged_cl_array(ary, axes=None, tags=frozenset())
            else:
                raise TypeError(f"{type(self).__name__}.thaw expects "
                                "'TaggableCLArray' or 'cl.array.Array' got "
                                f"{type(ary)}.")
            return pt.make_data_wrapper(ary.with_queue(self.queue),
                                        axes=get_pt_axes_from_cl_axes(ary.axes),
                                        tags=ary.tags)

        return with_array_context(rec_map_array_container(_rec_thaw, array),
                                  actx=self)

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
        return rec_map_array_container(
                lambda x: x.tagged(_preprocess_array_tags(tags)),
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
            ]).tagged(_preprocess_array_tags(tagged))

# }}}


# {{{ PytatoJAXArrayContext

class PytatoJAXArrayContext(_BasePytatoArrayContext):
    """
    An arraycontext that uses :mod:`pytato` to represent the thawed state of
    the arrays and compiles the expressions using
    :class:`pytato.target.python.JAXPythonTarget`.
    """

    def __init__(self,
            *, compile_trace_callback: Optional[Callable[[Any, str, Any], None]]
             = None) -> None:
        """
        :arg compile_trace_callback: A function of three arguments
            *(what, stage, ir)*, where *what* identifies the object
            being compiled, *stage* is a string describing the compilation
            pass, and *ir* is an object containing the intermediate
            representation. This interface should be considered
            unstable.
        """
        import pytato as pt
        from jax.numpy import DeviceArray
        super().__init__(compile_trace_callback=compile_trace_callback)
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
        from arraycontext.container import ArrayT
        from arraycontext.container.traversal import rec_keyed_map_array_container
        from arraycontext.impl.pytato.compile import _ary_container_key_stringifier

        array_as_dict: Dict[str, Union[DeviceArray, pt.Array]] = {}
        key_to_frozen_subary: Dict[str, DeviceArray] = {}
        key_to_pt_arrays: Dict[str, pt.Array] = {}

        def _record_leaf_ary_in_dict(key: Tuple[Any, ...],
                                     ary: Union[DeviceArray, pt.Array]):
            key_str = "_ary" + _ary_container_key_stringifier(key)
            array_as_dict[key_str] = ary
            return ary

        rec_keyed_map_array_container(_record_leaf_ary_in_dict, array)

        # {{{ remove any non pytato arrays from array_as_dict

        for key, subary in array_as_dict.items():
            if isinstance(subary, DeviceArray):
                key_to_frozen_subary[key] = subary.block_until_ready()
            elif isinstance(subary, pt.DataWrapper):
                # trivial freeze.
                key_to_frozen_subary[key] = subary.data.block_until_ready()
            else:
                if not isinstance(subary, pt.Array):
                    raise TypeError(f"{type(self).__name__}.freeze invoked "
                                    f"with non-pytato array of type '{type(array)}'")

                key_to_pt_arrays[key] = subary

        # }}}

        pt_dict_of_named_arrays = pt.make_dict_of_named_arrays(key_to_pt_arrays)
        transformed_dag = self.transform_dag(pt_dict_of_named_arrays)
        pt_prg = pt.generate_jax(transformed_dag, jit=True)
        out_dict = pt_prg()
        assert len(set(out_dict) & set(key_to_frozen_subary)) == 0

        key_to_frozen_subary = {
            **key_to_frozen_subary,
            **{k: v.block_until_ready()
               for k, v in out_dict.items()}
        }

        def _to_frozen(key: Tuple[Any, ...], ary: ArrayT):
            key_str = "_ary" + _ary_container_key_stringifier(key)
            return key_to_frozen_subary[key_str]

        return with_array_context(rec_keyed_map_array_container(_to_frozen,
                                                                array),
                                  actx=None)

    def thaw(self, array):
        import pytato as pt
        from jax.numpy import DeviceArray

        def _rec_thaw(ary):
            if isinstance(ary, DeviceArray):
                pass
            else:
                raise TypeError(f"{type(self).__name__}.thaw expects "
                                f"'jax.DeviceArray' got {type(ary)}.")
            return pt.make_data_wrapper(ary)

        return with_array_context(rec_map_array_container(_rec_thaw, array),
                                  actx=self)

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
                return ary.tagged(_preprocess_array_tags(tags))

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
            ]).tagged(_preprocess_array_tags(tagged))

# }}}


# vim: foldmethod=marker
