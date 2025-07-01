from __future__ import annotations


__doc__ = """
.. currentmodule:: arraycontext

A :mod:`pytato`-based array context defers the evaluation of an array until it is
frozen. The execution contexts for the evaluations are specific to an
:class:`~arraycontext.ArrayContext` type. For example,
:class:`~arraycontext.PytatoPyOpenCLArrayContext` uses :mod:`pyopencl` to
JIT-compile and execute the array expressions.

The following :mod:`pytato`-based array contexts are provided:

.. autoclass:: PytatoPyOpenCLArrayContext
.. autoclass:: PytatoJAXArrayContext


Compiling a Python callable (Internal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: arraycontext.impl.pytato.compile


Utils
^^^^^

.. automodule:: arraycontext.impl.pytato.utils
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

import abc
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from typing_extensions import override

from pytools import memoize_method
from pytools.tag import Tag, ToTagSetConvertible, normalize_tags

from arraycontext.container.traversal import (
    rec_map_container,
    with_array_context,
)
from arraycontext.context import (
    Array,
    ArrayContext,
    ArrayOrContainerOrScalarT,
    ArrayOrContainerT,
    ArrayOrScalar,
    P,
    ScalarLike,
    UntransformedCodeWarning,
    is_scalar_like,
)
from arraycontext.metadata import NameHint


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping

    import jax.numpy as jnp
    import loopy as lp
    import pyopencl as cl
    import pyopencl.array as cl_array
    import pytato
    import pytato as pt
    from loopy import TranslationUnit

    from arraycontext.container import SerializationKey

if getattr(sys, "_BUILDING_SPHINX_DOCS", False):
    pass

import logging


logger = logging.getLogger(__name__)

_EMPTY_TAG_SET: frozenset[Tag] = frozenset()


# {{{ tag conversion

def _preprocess_array_tags(tags: ToTagSetConvertible) -> frozenset[Tag]:
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
                    "was already present.", stacklevel=1)

        tags = (
                (tags | frozenset({PrefixNamed(name_hint.name)}))
                - {name_hint})

    return tags

# }}}


class _NotOnlyDataWrappers(Exception):  # noqa: N818
    pass


# {{{ _BasePytatoArrayContext

class _BasePytatoArrayContext(ArrayContext, abc.ABC):
    """
    An abstract :class:`ArrayContext` that uses :mod:`pytato` data types to
    represent.

    .. automethod:: __init__

    .. automethod:: transform_dag

    .. automethod:: compile
    """

    def __init__(
            self, *,
            compile_trace_callback: Callable[[Any, str, Any], None] | None = None
            ) -> None:
        """
        :arg compile_trace_callback: A function of three arguments
            *(what, stage, ir)*, where *what* identifies the object
            being compiled, *stage* is a string describing the compilation
            pass, and *ir* is an object containing the intermediate
            representation. This interface should be considered
            unstable.
        """
        super().__init__()

        self._freeze_prg_cache: dict[
            pt.AbstractResultWithNamedArrays, lp.TranslationUnit] = {}
        self._dag_transform_cache: dict[
                pt.AbstractResultWithNamedArrays,
                tuple[pt.AbstractResultWithNamedArrays, str]] = {}

        if compile_trace_callback is None:
            def _compile_trace_callback(what, stage, ir):
                pass

            compile_trace_callback = _compile_trace_callback

        self._compile_trace_callback = compile_trace_callback

    def _get_fake_numpy_namespace(self):
        from arraycontext.impl.pytato.fake_numpy import PytatoFakeNumpyNamespace
        return PytatoFakeNumpyNamespace(self)

    @property
    @abc.abstractmethod
    def _frozen_array_types(self) -> tuple[type, ...]:
        """
        Returns valid frozen array types for the array context.
        """

    def _rec_map_container(self,
               func: Callable[[Array], Array],
               array: ArrayOrContainerOrScalarT,
               allowed_types: tuple[type, ...] | None = None, *,
               default_scalar: ScalarLike | None = None,
            ) -> ArrayOrContainerOrScalarT:
        if allowed_types is None:
            allowed_types = self.array_types

        def _wrapper(ary: ArrayOrScalar) -> ArrayOrScalar:
            if isinstance(ary, allowed_types):
                return func(cast("Array", ary))
            elif is_scalar_like(ary):
                if default_scalar is None:
                    return ary
                else:
                    return np.array(ary).dtype.type(default_scalar)
            else:
                raise TypeError(
                    f"{type(self).__name__}.{func.__name__[1:]} invoked with "
                    f"an unsupported array type: got '{type(ary).__name__}', "
                    f"but expected one of {allowed_types}")

        return cast(
            "ArrayOrContainerOrScalarT",
            rec_map_container(_wrapper, array))

    @override
    def tag_axis(self,
                 iaxis: int, tags: ToTagSetConvertible,
                 array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        def _tag_axis(ary: ArrayOrScalar) -> ArrayOrScalar:
            return cast("pt.Array", ary).with_tagged_axis(iaxis, tags)

        return cast("ArrayOrContainerOrScalarT",
                    rec_map_container(_tag_axis, array))

    # {{{ compilation

    def transform_dag(self, dag: pytato.AbstractResultWithNamedArrays
                      ) -> pytato.AbstractResultWithNamedArrays:
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

    def transform_loopy_program(self, t_unit: lp.TranslationUnit) -> lp.TranslationUnit:
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

    # }}}

    # {{{ properties

    @property
    def permits_inplace_modification(self):
        return False

    @property
    def supports_nonscalar_broadcasting(self):
        return True

    @property
    def permits_advanced_indexing(self):
        return True

    def get_target(self):
        return None

    # }}}

    @override
    def outline(self,
                f: Callable[P, ArrayOrContainerOrScalarT],
                *,
                id: Hashable | None = None,
                tags: frozenset[Tag] = _EMPTY_TAG_SET,
            ) -> Callable[P, ArrayOrContainerOrScalarT]:
        from pytato.tags import FunctionIdentifier

        from .outline import OutlinedCall
        id = id or getattr(f, "__name__", None)
        if id is not None:
            tags = tags | {FunctionIdentifier(id)}

        # FIXME Ideally, the ParamSpec P should be bounded by ArrayOrContainerOrScalar,
        # but this is not currently possible:
        # https://github.com/python/typing/issues/1027

        # FIXME An aspect of this that's a bit of a lie is that the types
        # coming out of the outlined function are not guaranteed to be the same
        # as the ones that the un-outlined function would return. That said,
        # if f is written only in terms of the array context types (Array, ScalarLike,
        # containers), this is close enough to being true that I'm willing
        # to take responsibility. -AK, 2025-06-30
        return cast("Callable[P, ArrayOrContainerOrScalarT]",
                    cast("object", OutlinedCall(self, f, tags)))

# }}}


# {{{ PytatoPyOpenCLArrayContext


@dataclass
class ProfileEvent:
    """Holds a profile event that has not been collected by the profiler yet."""

    start_cl_event: cl.Event
    stop_cl_event: cl.Event
    t_unit_name: str


class PytatoPyOpenCLArrayContext(_BasePytatoArrayContext):
    """
    An :class:`ArrayContext` that uses :mod:`pytato` data types to represent
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
    context: cl.Context
    queue: cl.CommandQueue
    allocator: cl_array.Allocator
    using_svm: bool | None
    profile_kernels: bool

    _force_svm_arg_limit: int | None

    def __init__(
            self, queue: cl.CommandQueue,
            allocator: cl_array.Allocator | None = None,
            *,
            use_memory_pool: bool | None = None,
            compile_trace_callback: Callable[[Any, str, Any], None] | None = None,
            profile_kernels: bool = False,
            # do not use: only for testing
            _force_svm_arg_limit: int | None = None,
            ) -> None:
        """
        :arg compile_trace_callback: A function of three arguments
            *(what, stage, ir)*, where *what* identifies the object
            being compiled, *stage* is a string describing the compilation
            pass, and *ir* is an object containing the intermediate
            representation. This interface should be considered
            unstable.
        """
        if allocator is not None and use_memory_pool is not None:
            raise TypeError("may not specify both allocator and use_memory_pool")

        self.using_svm = None

        if allocator is None:
            from pyopencl.characterize import has_coarse_grain_buffer_svm
            has_svm = has_coarse_grain_buffer_svm(queue.device)
            if has_svm:
                self.using_svm = True

                from pyopencl.tools import SVMAllocator
                allocator = SVMAllocator(queue.context, queue=queue)

                if use_memory_pool:
                    from pyopencl.tools import SVMPool
                    allocator = SVMPool(allocator)
            else:
                self.using_svm = False

                from pyopencl.tools import ImmediateAllocator
                allocator = ImmediateAllocator(queue)

                if use_memory_pool:
                    from pyopencl.tools import MemoryPool
                    allocator = MemoryPool(allocator)
        else:
            # Check whether the passed allocator allocates SVM
            try:
                from pyopencl import SVMPointer
                mem = allocator(4)
                if isinstance(mem, SVMPointer):
                    self.using_svm = True
                else:
                    self.using_svm = False
            except ImportError:
                self.using_svm = False

        import pyopencl.array as cla
        import pytato as pt
        super().__init__(compile_trace_callback=compile_trace_callback)
        self.queue = queue

        self.allocator = allocator
        self.array_types = (pt.Array, cla.Array)

        # unused, but necessary to keep the context alive
        self.context = self.queue.context

        self._force_svm_arg_limit = _force_svm_arg_limit

        self._enable_profiling(profile_kernels)

    # {{{ Profiling functionality

    def _enable_profiling(self, enable: bool) -> None:
        # List of ProfileEvents that haven't been transferred to profiled
        # results yet
        self._profile_events: list[ProfileEvent] = []

        # Dict of kernel name -> list of kernel execution times
        self._profile_results: dict[str, list[int]] = {}

        if enable:
            import pyopencl as cl
            if not self.queue.properties & cl.command_queue_properties.PROFILING_ENABLE:
                raise RuntimeError("Profiling was not enabled in the command queue. "
                    "Please create the queue with "
                    "cl.command_queue_properties.PROFILING_ENABLE.")
            self.profile_kernels = True

        else:
            self.profile_kernels = False

    def _wait_and_transfer_profile_events(self) -> None:
        """Wait for all profiling events to finish and transfer the results
        to *self._profile_results*."""
        import pyopencl as cl
        # First, wait for completion of all events
        if self._profile_events:
            cl.wait_for_events([p_event.stop_cl_event
                                    for p_event in self._profile_events])

        # Then, collect all events and store them
        for t in self._profile_events:
            name = t.t_unit_name

            time = t.stop_cl_event.profile.end - t.start_cl_event.profile.end

            self._profile_results.setdefault(name, []).append(time)

        self._profile_events = []

    def _add_profiling_events(self, start: cl._cl.Event, stop: cl._cl.Event,
                             t_unit_name: str) -> None:
        """Add profiling events to the list of profiling events."""
        self._profile_events.append(ProfileEvent(start, stop, t_unit_name))

    def _reset_profiling_data(self) -> None:
        """Reset profiling data."""
        self._profile_results = {}

    # }}}

    @property
    def _frozen_array_types(self) -> tuple[type, ...]:
        import pyopencl.array as cla
        return (cla.Array,)

    # {{{ ArrayContext interface

    def from_numpy(self, array):
        import pytato as pt

        import arraycontext.impl.pyopencl.taggable_cl_array as tga

        def _from_numpy(ary: np.ndarray[Any, Any]) -> pt.Array:
            return pt.make_data_wrapper(
                tga.to_device(self.queue, ary, allocator=self.allocator)
                )

        return with_array_context(
            self._rec_map_container(_from_numpy, array, (np.ndarray,)),
            actx=self)

    def to_numpy(self, array):
        def _to_numpy(ary):
            return ary.get(queue=self.queue)

        return with_array_context(
            self._rec_map_container(_to_numpy, self.freeze(array)),
            actx=None)

    @memoize_method
    def get_target(self):
        import pyopencl as cl
        import pyopencl.characterize as cl_char

        dev = self.queue.device

        if (
                self._force_svm_arg_limit is not None
                or (
                    self.using_svm and dev.type & cl.device_type.GPU
                    and cl_char.has_coarse_grain_buffer_svm(dev))):

            if dev.max_parameter_size == 4352:
                # Nvidia devices and PTXAS declare a limit of 4352 bytes,
                # which is incorrect. The CUDA documentation at
                # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-parameters
                # mentions a limit of 4KB, which is also incorrect.
                # As far as I can tell, the actual limit is around 4080
                # bytes, at least on a K40. Reducing the limit further
                # in order to be on the safe side.

                # Note that the naming convention isn't super consistent
                # for Nvidia GPUs, so that we only use the maximum
                # parameter size to determine if it is an Nvidia GPU.

                limit = 4096-200

                from warnings import warn
                warn("Running on an Nvidia GPU, reducing the argument "
                    f"size limit from 4352 to {limit}.", stacklevel=1)
            else:
                limit = dev.max_parameter_size

            if self._force_svm_arg_limit is not None:
                limit = self._force_svm_arg_limit

            logger.info(
                    "limiting argument buffer size for %s to %d bytes",
                    dev, limit)

            from arraycontext.impl.pytato.utils import (
                ArgSizeLimitingPytatoLoopyPyOpenCLTarget,
            )
            return ArgSizeLimitingPytatoLoopyPyOpenCLTarget(limit)
        else:
            return super().get_target()

    @override
    def freeze(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        if np.isscalar(array):
            return array

        import pyopencl.array as cla
        import pytato as pt

        from arraycontext.container.traversal import rec_keyed_map_array_container
        from arraycontext.impl.pyopencl.taggable_cl_array import (
            TaggableCLArray,
            to_tagged_cl_array,
        )
        from arraycontext.impl.pytato.utils import (
            _ary_container_key_stringifier,
            _normalize_pt_expr,
            get_cl_axes_from_pt_axes,
        )

        array_as_dict: dict[str, cla.Array | TaggableCLArray | pt.Array] = {}
        key_to_frozen_subary: dict[str, TaggableCLArray] = {}
        key_to_pt_arrays: dict[str, pt.Array] = {}

        def _record_leaf_ary_in_dict(
                key: tuple[SerializationKey, ...],
                ary: ArrayOrScalar) -> ArrayOrScalar:
            key_str = "_ary" + _ary_container_key_stringifier(key)
            if not isinstance(ary, cla.Array | TaggableCLArray | pt.Array):
                raise TypeError(f"expected one of array_types, got {type(ary)}")
            array_as_dict[key_str] = ary
            return ary

        rec_keyed_map_array_container(_record_leaf_ary_in_dict, array)

        # {{{ remove any non pytato arrays from array_as_dict

        for key, subary in array_as_dict.items():
            if isinstance(subary, TaggableCLArray):
                key_to_frozen_subary[key] = subary.with_queue(None)
            elif isinstance(subary, self._frozen_array_types):
                from warnings import warn
                warn(f"Invoking {type(self).__name__}.freeze with"
                    f" {type(subary).__name__} will be unsupported in 2023. Use"
                    " `to_tagged_cl_array` to convert instances to TaggableCLArray.",
                    DeprecationWarning, stacklevel=2)

                key_to_frozen_subary[key] = (
                    to_tagged_cl_array(subary.with_queue(None)))
            elif isinstance(subary, pt.DataWrapper):
                # trivial freeze.
                key_to_frozen_subary[key] = to_tagged_cl_array(
                    subary.data,
                    axes=get_cl_axes_from_pt_axes(subary.axes),
                    tags=subary.tags)
            elif isinstance(subary, pt.Array):
                # Don't be tempted to take shortcuts here, e.g. for empty
                # arrays, as this will inhibit metadata propagation that
                # may happen in transform_dag below. See
                # https://github.com/inducer/arraycontext/pull/167#issuecomment-1151877480
                key_to_pt_arrays[key] = subary
            else:
                raise TypeError(
                    f"{type(self).__name__}.freeze invoked with an unsupported "
                    f"array type: got '{type(subary).__name__}', but expected one "
                    f"of {self.array_types}")

        # }}}

        def _to_frozen(
                    key: tuple[SerializationKey, ...],
                    ary: ArrayOrScalar
                ) -> ArrayOrScalar:
            key_str = "_ary" + _ary_container_key_stringifier(key)
            return key_to_frozen_subary[key_str]

        if not key_to_pt_arrays:
            # all cl arrays => no need to perform any codegen
            return with_array_context(
                    rec_keyed_map_array_container(_to_frozen, array),
                    actx=None)

        dag = pt.transform.deduplicate(
            pt.make_dict_of_named_arrays(key_to_pt_arrays))

        # FIXME: Remove this if/when _normalize_pt_expr gets support for functions
        dag = pt.tag_all_calls_to_be_inlined(dag)
        dag = pt.inline_calls(dag)

        normalized_expr, bound_arguments = _normalize_pt_expr(dag)

        try:
            pt_prg = self._freeze_prg_cache[normalized_expr]
        except KeyError:
            try:
                transformed_dag, function_name = (
                        self._dag_transform_cache[normalized_expr])
            except KeyError:
                transformed_dag = self.transform_dag(normalized_expr)

                from pytato.tags import PrefixNamed
                name_hint_tags = []
                for subary in key_to_pt_arrays.values():
                    name_hint_tags.extend(subary.tags_of_type(PrefixNamed))

                from pytools import common_prefix
                name_hint = common_prefix([nh.prefix for nh in name_hint_tags])

                # All name_hint_tags shared at least some common prefix.
                function_name = f"frozen_{name_hint}" if name_hint else "frozen_result"

                self._dag_transform_cache[normalized_expr] = (
                        transformed_dag, function_name)

            from arraycontext.loopy import _DEFAULT_LOOPY_OPTIONS
            opts = _DEFAULT_LOOPY_OPTIONS
            assert opts.return_dict

            pt_prg = pt.generate_loopy(transformed_dag,
                                       options=opts,
                                       function_name=function_name,
                                       target=self.get_target()
                                       ).bind_to_context(self.context)
            pt_prg = pt_prg.with_transformed_translation_unit(
                    self.transform_loopy_program)
            self._freeze_prg_cache[normalized_expr] = pt_prg
        else:
            transformed_dag, function_name = (
                    self._dag_transform_cache[normalized_expr])

        assert len(pt_prg.bound_arguments) == 0

        if self.profile_kernels:
            import pyopencl as cl
            start_evt = cl.enqueue_marker(self.queue)

        evt, out_dict = pt_prg(self.queue,
                allocator=self.allocator,
                **bound_arguments)

        if self.profile_kernels:
            self._add_profiling_events(start_evt, evt, pt_prg.program.entrypoint)

        assert len(set(out_dict) & set(key_to_frozen_subary)) == 0

        key_to_frozen_subary = {
            **key_to_frozen_subary,
            **{k: to_tagged_cl_array(
                    v.with_queue(None),
                    axes=get_cl_axes_from_pt_axes(transformed_dag[k].expr.axes),
                    tags=transformed_dag[k].expr.tags)
               for k, v in out_dict.items()}
        }

        return with_array_context(
                rec_keyed_map_array_container(_to_frozen, array),
                actx=None)

    def thaw(self, array):
        import pytato as pt

        import arraycontext.impl.pyopencl.taggable_cl_array as tga
        from .utils import get_pt_axes_from_cl_axes

        def _thaw(ary: tga.TaggableCLArray) -> pt.Array:
            return pt.make_data_wrapper(
                ary.with_queue(self.queue),
                axes=get_pt_axes_from_cl_axes(ary.axes),
                tags=ary.tags)

        return with_array_context(
            self._rec_map_container(_thaw, array, (tga.TaggableCLArray,)),
            actx=self)

    def freeze_thaw(self, array):
        import pytato as pt

        import arraycontext.impl.pyopencl.taggable_cl_array as tga

        def _ft(ary: tga.TaggableCLArray | pt.Array) -> tga.TaggableCLArray | pt.Array:
            if isinstance(ary, (pt.DataWrapper, tga.TaggableCLArray)):
                return ary
            else:
                raise _NotOnlyDataWrappers()

        try:
            return with_array_context(
                self._rec_map_container(_ft, array),
                actx=self)
        except _NotOnlyDataWrappers:
            return super().freeze_thaw(array)

    def tag(self, tags: ToTagSetConvertible, array):
        def _tag(ary):
            return ary.tagged(_preprocess_array_tags(tags))

        return self._rec_map_container(_tag, array)

    # }}}

    # {{{ compilation

    def call_loopy(self, program, **kwargs):
        import pytato as pt
        from pytato.loopy import call_loopy
        from pytato.scalar_expr import SCALAR_CLASSES

        from arraycontext.impl.pyopencl.taggable_cl_array import TaggableCLArray

        entrypoint = program.default_entrypoint.name

        # {{{ preprocess args

        processed_kwargs = {}

        for kw, arg in sorted(kwargs.items()):
            if isinstance(arg, (pt.Array, *SCALAR_CLASSES)):
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

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        from .compile import LazilyPyOpenCLCompilingFunctionCaller
        return LazilyPyOpenCLCompilingFunctionCaller(self, f)

    def transform_dag(self, dag: pytato.AbstractResultWithNamedArrays
                      ) -> pytato.AbstractResultWithNamedArrays:
        import pytato as pt
        dag = pt.tag_all_calls_to_be_inlined(dag)
        dag = pt.inline_calls(dag)
        dag = pt.transform.materialize_with_mpms(dag)
        return dag

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        import pytato as pt

        import arraycontext.impl.pyopencl.taggable_cl_array as tga

        if arg_names is None:
            arg_names = (None,) * len(args)

        def preprocess_arg(name, arg):
            if isinstance(arg, tga.TaggableCLArray):
                ary = self.thaw(arg)
            elif isinstance(arg, self._frozen_array_types):
                from warnings import warn
                warn(f"Invoking {type(self).__name__}.einsum with"
                    f" {type(arg).__name__} will be unsupported in 2023. Use"
                    " `to_tagged_cl_array` to convert instances to TaggableCLArray.",
                    DeprecationWarning, stacklevel=2)
                ary = self.thaw(tga.to_tagged_cl_array(arg))
            elif isinstance(arg, pt.Array):
                ary = arg
            else:
                raise TypeError(
                    f"{type(self).__name__}.einsum invoked with an unsupported "
                    f"array type: got '{type(arg).__name__}', but expected one "
                    f"of {self.array_types}")

            if name is not None:  # noqa: SIM102
                # Tagging Placeholders with naming-related tags is pointless:
                # They already have names. It's also counterproductive, as
                # multiple placeholders with the same name that are not
                # also the same object are not allowed, and this would produce
                # a different Placeholder object of the same name.
                if (not isinstance(ary, pt.Placeholder | pt.NamedArray)
                        and not ary.tags_of_type(NameHint)):
                    ary = ary.tagged(NameHint(name))

            return ary

        return pt.einsum(spec, *[
            preprocess_arg(name, arg)
            for name, arg in zip(arg_names, args, strict=True)
            ]).tagged(_preprocess_array_tags(tagged))

    def clone(self):
        return type(self)(self.queue, self.allocator)

    # }}}

# }}}


# {{{ PytatoJAXArrayContext

class PytatoJAXArrayContext(_BasePytatoArrayContext):
    """
    An arraycontext that uses :mod:`pytato` to represent the thawed state of
    the arrays and compiles the expressions using
    :class:`pytato.target.python.JAXPythonTarget`.

    .. automethod:: transform_dag
    """

    def __init__(self,
            *,
            compile_trace_callback: Callable[[Any, str, Any], None] | None = None,
            ) -> None:
        """
        :arg compile_trace_callback: A function of three arguments
            *(what, stage, ir)*, where *what* identifies the object
            being compiled, *stage* is a string describing the compilation
            pass, and *ir* is an object containing the intermediate
            representation. This interface should be considered
            unstable.
        """
        import jax.numpy as jnp
        import pytato as pt
        super().__init__(compile_trace_callback=compile_trace_callback)
        self.array_types = (pt.Array, jnp.ndarray)

    @property
    def _frozen_array_types(self) -> tuple[type, ...]:
        import jax.numpy as jnp
        return (jnp.ndarray, )

    # {{{ ArrayContext interface

    def from_numpy(self, array):
        import jax
        import pytato as pt

        def _from_numpy(ary: np.ndarray[Any, Any]) -> pt.Array:
            return pt.make_data_wrapper(jax.device_put(ary))

        return with_array_context(
            self._rec_map_container(_from_numpy, array, (np.ndarray,)),
            actx=self)

    def to_numpy(self, array):
        import jax

        def _to_numpy(ary):
            return jax.device_get(ary)

        return with_array_context(
            self._rec_map_container(_to_numpy, self.freeze(array)),
            actx=None)

    @override
    def freeze(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        if np.isscalar(array):
            return array

        import jax.numpy as jnp
        import pytato as pt

        from arraycontext.container.traversal import rec_keyed_map_array_container
        from arraycontext.impl.pytato.utils import _ary_container_key_stringifier

        array_as_dict: dict[str, jnp.ndarray | pt.Array] = {}
        key_to_frozen_subary: dict[str, jnp.ndarray] = {}
        key_to_pt_arrays: dict[str, pt.Array] = {}

        def _record_leaf_ary_in_dict(key: tuple[SerializationKey, ...],
                                     ary: ArrayOrScalar) -> ArrayOrScalar:
            key_str = "_ary" + _ary_container_key_stringifier(key)
            array_as_dict[key_str] = cast("jnp.ndarray", cast("object", ary))
            return ary

        rec_keyed_map_array_container(_record_leaf_ary_in_dict, array)

        # {{{ remove any non pytato arrays from array_as_dict

        for key, subary in array_as_dict.items():
            if isinstance(subary, jnp.ndarray):
                key_to_frozen_subary[key] = subary.block_until_ready()
            elif isinstance(subary, pt.DataWrapper):
                # trivial freeze.
                key_to_frozen_subary[key] = subary.data.block_until_ready()
            elif isinstance(subary, pt.Array):
                key_to_pt_arrays[key] = subary
            else:
                raise TypeError(
                    f"{type(self).__name__}.freeze invoked with an unsupported "
                    f"array type: got '{type(subary).__name__}', but expected one "
                    f"of {self.array_types}")

        # }}}

        def _to_frozen(
                    key: tuple[SerializationKey, ...],
                    ary: ArrayOrScalar,  # pyright: ignore[reportUnusedParameter]
                ) -> ArrayOrScalar:
            key_str = "_ary" + _ary_container_key_stringifier(key)
            return cast("Array", cast("object", key_to_frozen_subary[key_str]))

        if not key_to_pt_arrays:
            # all cl arrays => no need to perform any codegen
            return with_array_context(
                    rec_keyed_map_array_container(_to_frozen, array),
                    actx=None)

        pt_dict_of_named_arrays = pt.transform.deduplicate(
            pt.make_dict_of_named_arrays(key_to_pt_arrays))

        transformed_dag = self.transform_dag(pt_dict_of_named_arrays)
        pt_prg = pt.generate_jax(transformed_dag, jit=True)
        out_dict = pt_prg()
        assert len(set(out_dict) & set(key_to_frozen_subary)) == 0

        key_to_frozen_subary = {
            **key_to_frozen_subary,
            **{k: v.block_until_ready()
               for k, v in out_dict.items()}
        }

        return with_array_context(
            rec_keyed_map_array_container(_to_frozen, array),
            actx=None)

    @override
    def thaw(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        import pytato as pt

        def _thaw(ary: jnp.ndarray) -> pt.Array:
            return pt.make_data_wrapper(ary)

        return with_array_context(
            self._rec_map_container(_thaw, array, self._frozen_array_types),
            actx=self)

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        from .compile import LazilyJAXCompilingFunctionCaller
        return LazilyJAXCompilingFunctionCaller(self, f)

    @override
    def transform_dag(self, dag: pytato.AbstractResultWithNamedArrays
                      ) -> pytato.AbstractResultWithNamedArrays:
        import pytato as pt
        dag = pt.tag_all_calls_to_be_inlined(dag)
        dag = pt.inline_calls(dag)
        return dag

    @override
    def tag(self,
                tags: ToTagSetConvertible,
                array: ArrayOrContainerT,
            ) -> ArrayOrContainerT:
        def _tag(ary: Array) -> Array:
            import jax.numpy as jnp
            if isinstance(ary, jnp.ndarray):
                return ary
            else:
                return cast("pt.Array", ary).tagged(_preprocess_array_tags(tags))

        return self._rec_map_container(_tag, array)

    # }}}

    # {{{ compilation

    @override
    def call_loopy(self,
                   t_unit: TranslationUnit,
                   **kwargs: Any) -> Mapping[str, Array]:
        raise NotImplementedError(
            "Calling loopy on JAX arrays is not supported. Maybe rewrite"
            " the loopy kernel as numpy-flavored array operations using"
            " ArrayContext.np.")

    @override
    def einsum(self,
               spec: str, *args: Array,
               arg_names: tuple[str | None, ...] | None = None,
               tagged: ToTagSetConvertible = ()) -> Array:
        import pytato as pt
        if arg_names is None:
            arg_names = (None,) * len(args)

        def preprocess_arg(name: str | None, arg: Array):
            import jax.numpy as jnp
            if isinstance(arg, jnp.ndarray):
                ary = cast("pt.Array", cast("object", self.thaw(arg)))
            elif isinstance(arg, pt.Array):
                ary = arg
            else:
                raise TypeError(
                    f"{type(self).__name__}.einsum invoked with an unsupported "
                    f"array type: got '{type(arg).__name__}', but expected one "
                    f"of {self.array_types}")

            if name is not None:  # noqa: SIM102
                # Tagging Placeholders with naming-related tags is pointless:
                # They already have names. It's also counterproductive, as
                # multiple placeholders with the same name that are not
                # also the same object are not allowed, and this would produce
                # a different Placeholder object of the same name.
                if (not isinstance(ary, pt.Placeholder)
                        and not ary.tags_of_type(NameHint)):
                    ary = ary.tagged(NameHint(name))

            return ary

        return cast("pt.Array", pt.einsum(spec, *[
            preprocess_arg(name, arg)
            for name, arg in zip(arg_names, args, strict=True)
            ]).tagged(_preprocess_array_tags(tagged)))

    @override
    def clone(self):
        return type(self)()

# }}}

# }}}

# vim: foldmethod=marker
