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

import abc
import sys
from typing import (  # noqa
    Any, Callable, Dict, FrozenSet, Tuple, Type, Union, TypeVar, Optional,
    Hashable, Sequence, ClassVar, Iterator, Iterable, Mapping,
    TYPE_CHECKING
)
import numpy as np

from arraycontext.context import (
    ArrayT, ArrayContext, Array, ArrayOrContainer, ScalarLike
)
from arraycontext.container.traversal import (
    rec_map_array_container,
    with_array_context,
    rec_keyed_map_array_container
)

from arraycontext.container import ArrayContainer, is_array_container_type
from arraycontext.metadata import NameHint
from pytools.tag import Tag, ToTagSetConvertible, normalize_tags
from pytools import memoize_method
from dataclasses import dataclass
from pyrsistent import pmap, PMap
import pytato as pt
# from pt.array import _get_default_axes, _get_default_tags
# from pt.tags import FunctionIdentifier
import itertools

if TYPE_CHECKING:
    # import pytato
    import pyopencl as cl
    import pytato

if getattr(sys, "_BUILDING_SPHINX_DOCS", False):
    import pyopencl as cl  # noqa: F811

import re
import logging


logger = logging.getLogger(__name__)

ReturnT = TypeVar("ReturnT", Array, Tuple[Array, ...], Dict[str, Array],
                  ArrayContainer)
RE_ARGNAME = re.compile(r"^_pt_(\d+)$")


def _to_identifier(s: str) -> str:
    return "".join(ch for ch in s if ch.isidentifier())


def _prg_id_to_kernel_name(f: Any) -> str:
    if callable(f):
        name = getattr(f, "__name__", "<anonymous>")
        if not name.isidentifier():
            return "actx_compiled_" + _to_identifier(name)
        else:
            return name
    else:
        return _to_identifier(str(f))


class _Guess():
    pass


class FromArrayContextCompile(Tag):
    """
    Tagged to the entrypoint kernel of every translation unit that is generated
    by :meth:`~arraycontext.PytatoPyOpenCLArrayContext.compile`.

    Typically this tag serves as a branch condition in implementing a
    specialized transform strategy for kernels compiled by
    :meth:`~arraycontext.PytatoPyOpenCLArrayContext.compile`.
    """


# {{{ helper classes: AbstractInputDescriptor

class AbstractInputDescriptor:
    """
    Used internally in :class:`BaseLazilyCompilingFunctionCaller` to characterize
    an input.
    """
    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError


@dataclass(frozen=True, eq=True)
class ScalarInputDescriptor(AbstractInputDescriptor):
    dtype: np.dtype


@dataclass(frozen=True, eq=True)
class LeafArrayDescriptor(AbstractInputDescriptor):
    dtype: np.dtype
    shape: pt.array.ShapeType

# }}}


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

    def __init__(
            self, *,
            compile_trace_callback: Optional[Callable[[Any, str, Any], None]] = None
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

        import loopy as lp
        import pytato as pt
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

    @abc.abstractproperty
    def _frozen_array_types(self) -> Tuple[Type, ...]:
        """
        Returns valid frozen array types for the array context.
        """

    # {{{ ArrayContext interface

    def empty(self, shape, dtype):
        raise NotImplementedError(
            f"{type(self).__name__}.empty is not supported")

    def zeros(self, shape, dtype):
        import pytato as pt
        return pt.zeros(shape, dtype)

    def empty_like(self, ary):
        raise NotImplementedError(
            f"{type(self).__name__}.empty_like is not supported")

    # }}}

    # {{{ compilation

    def transform_dag(self, dag: "pt.DictOfNamedArrays"
                      ) -> "pt.DictOfNamedArrays":
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
        raise ValueError(
            f"{type(self).__name__} does not implement transform_loopy_program. "
            "Sub-classes are supposed to implement it.")

    @abc.abstractmethod
    def einsum(self, spec, *args, arg_names=None, tagged=()):
        pass

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
    def __init__(
            self, queue: "cl.CommandQueue", allocator=None, *,
            use_memory_pool: Optional[bool] = None,
            compile_trace_callback: Optional[Callable[[Any, str, Any], None]] = None,

            # do not use: only for testing
            _force_svm_arg_limit: Optional[int] = None,
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

    @property
    def _frozen_array_types(self) -> Tuple[Type, ...]:
        import pyopencl.array as cla
        return (cla.Array,)

    def _rec_map_container(
            self, func: Callable[[Array], Array], array: ArrayOrContainer,
            allowed_types: Optional[Tuple[type, ...]] = None, *,
            default_scalar: Optional[ScalarLike] = None,
            strict: bool = False) -> ArrayOrContainer:
        import pytato as pt

        import arraycontext.impl.pyopencl.taggable_cl_array as tga

        if allowed_types is None:
            allowed_types = (pt.Array, tga.TaggableCLArray)

        def _wrapper(ary):
            if isinstance(ary, allowed_types):
                return func(ary)
            elif not strict and isinstance(ary, self._frozen_array_types):
                from warnings import warn
                warn(f"Invoking {type(self).__name__}.{func.__name__[1:]} with"
                    f" {type(ary).__name__} will be unsupported in 2023. Use"
                    " 'to_tagged_cl_array' to convert instances to"
                    " TaggableCLArray.", DeprecationWarning, stacklevel=2)

                return func(tga.to_tagged_cl_array(ary))
            elif np.isscalar(ary):
                if default_scalar is None:
                    return ary
                else:
                    return np.array(ary).dtype.type(default_scalar)
            else:
                raise TypeError(
                    f"{type(self).__name__}.{func.__name__[1:]} invoked with "
                    f"an unsupported array type: got '{type(ary).__name__}', "
                    f"but expected one of {allowed_types}")

        return rec_map_array_container(_wrapper, array)

    # {{{ ArrayContext interface

    def zeros_like(self, ary):
        from warnings import warn
        warn(f"{type(self).__name__}.zeros_like is deprecated and will stop "
            "working in 2023. Use actx.np.zeros_like instead.",
            DeprecationWarning, stacklevel=2)

        return self.np.zeros_like(ary)

    def from_numpy(self, array):
        import pytato as pt

        import arraycontext.impl.pyopencl.taggable_cl_array as tga

        def _from_numpy(ary):
            return pt.make_data_wrapper(
                tga.to_device(self.queue, ary, allocator=self.allocator)
                )

        return with_array_context(
            self._rec_map_container(_from_numpy, array, (np.ndarray,), strict=True),
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
                    f"size limit from 4352 to {limit}.")
            else:
                limit = dev.max_parameter_size

            if self._force_svm_arg_limit is not None:
                limit = self._force_svm_arg_limit

            logger.info(f"limiting argument buffer size for {dev} to {limit} bytes")

            from arraycontext.impl.pytato.utils import (
                ArgSizeLimitingPytatoLoopyPyOpenCLTarget)
            return ArgSizeLimitingPytatoLoopyPyOpenCLTarget(limit)
        else:
            return super().get_target()

    def freeze(self, array):
        if np.isscalar(array):
            return array

        import pyopencl.array as cla
        import pytato as pt

        from arraycontext.container.traversal import rec_keyed_map_array_container
        from arraycontext.impl.pyopencl.taggable_cl_array import (
            TaggableCLArray, to_tagged_cl_array)
        from arraycontext.impl.pytato.compile import _ary_container_key_stringifier
        from arraycontext.impl.pytato.utils import (
            _normalize_pt_expr, get_cl_axes_from_pt_axes)

        array_as_dict: Dict[str, Union[cla.Array, TaggableCLArray, pt.Array]] = {}
        key_to_frozen_subary: Dict[str, TaggableCLArray] = {}
        key_to_pt_arrays: Dict[str, pt.Array] = {}

        def _record_leaf_ary_in_dict(
                key: Tuple[Any, ...],
                ary: Union[cla.Array, TaggableCLArray, pt.Array]) -> None:
            key_str = "_ary" + _ary_container_key_stringifier(key)
            array_as_dict[key_str] = ary

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

        pt_dict_of_named_arrays = pt.make_dict_of_named_arrays(
                key_to_pt_arrays)
        normalized_expr, bound_arguments = _normalize_pt_expr(
                pt_dict_of_named_arrays)

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
                if name_hint:
                    # All name_hint_tags shared at least some common prefix.
                    function_name = f"frozen_{name_hint}"
                else:
                    function_name = "frozen_result"

                self._dag_transform_cache[normalized_expr] = (
                        transformed_dag, function_name)

            from arraycontext.loopy import _DEFAULT_LOOPY_OPTIONS
            pt_prg = pt.generate_loopy(transformed_dag,
                                       options=_DEFAULT_LOOPY_OPTIONS,
                                       cl_device=self.queue.device,
                                       function_name=function_name,
                                       target=self.get_target())
            pt_prg = pt_prg.with_transformed_program(self.transform_loopy_program)
            self._freeze_prg_cache[normalized_expr] = pt_prg
        else:
            transformed_dag, function_name = (
                    self._dag_transform_cache[normalized_expr])

        assert len(pt_prg.bound_arguments) == 0
        evt, out_dict = pt_prg(self.queue,
                allocator=self.allocator,
                **bound_arguments)
        evt.wait()
        assert len(set(out_dict) & set(key_to_frozen_subary)) == 0

        key_to_frozen_subary = {
            **key_to_frozen_subary,
            **{k: to_tagged_cl_array(
                    v.with_queue(None),
                    axes=get_cl_axes_from_pt_axes(transformed_dag[k].expr.axes),
                    tags=transformed_dag[k].expr.tags)
               for k, v in out_dict.items()}
        }

        def _to_frozen(key: Tuple[Any, ...], ary) -> TaggableCLArray:
            key_str = "_ary" + _ary_container_key_stringifier(key)
            return key_to_frozen_subary[key_str]

        return with_array_context(
                rec_keyed_map_array_container(_to_frozen, array),
                actx=None)

    def thaw(self, array):
        import pytato as pt

        import arraycontext.impl.pyopencl.taggable_cl_array as tga
        from .utils import get_pt_axes_from_cl_axes

        def _thaw(ary):
            return pt.make_data_wrapper(ary.with_queue(self.queue),
                                        axes=get_pt_axes_from_cl_axes(ary.axes),
                                        tags=ary.tags)

        return with_array_context(
            self._rec_map_container(_thaw, array, (tga.TaggableCLArray,)),
            actx=self)

    def tag(self, tags: ToTagSetConvertible, array):
        def _tag(ary):
            return ary.tagged(_preprocess_array_tags(tags))

        return self._rec_map_container(_tag, array)

    def tag_axis(self, iaxis, tags: ToTagSetConvertible, array):
        def _tag_axis(ary):
            return ary.with_tagged_axis(iaxis, tags)

        return self._rec_map_container(_tag_axis, array)

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

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        from .compile import LazilyPyOpenCLCompilingFunctionCaller
        return LazilyPyOpenCLCompilingFunctionCaller(self, f)

    def transform_dag(self, dag: "pt.DictOfNamedArrays"
                      ) -> "pt.DictOfNamedArrays":
        import pytato as pt
        dag = pt.transform.materialize_with_mpms(dag)
        dag = pt.tag_all_calls_to_be_inlined(dag)

        # concated_dag = \
        #    pt.concatenate_calls(
        #        dag, (lambda x: pt.tags.FunctionIdentifier("wvflux_int")
        #              in x.call.function.tags))

        # concated_dag = \
        #    pt.concatenate_calls(
        #        dag, (lambda x: True))

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

    def trace_call(self, f: Callable[..., ReturnT],
                   *args: Array,
                   identifier: Optional[Hashable] = None,
                   **kwargs: Array) -> ReturnT:
        """
        Returns the expressions returned after calling *f* with the arguments
        *args* and keyword arguments *kwargs*. The subexpressions in the returned
        expressions are outlined as a :class:`~pytato.tracing.FunctionDefinition`.

        :arg identifier: A hashable object that acts as
            :attr:`pytato.tags.FunctionIdentifier.identifier` for the
            :class:`~pytato.tags.FunctionIdentifier` tagged to the outlined
            :class:`~pytato.tracing.FunctionDefinition`. If ``None`` the function
            definition is not tagged with a
            :class:`~pytato.tags.FunctionIdentifier` tag, if ``_Guess`` the
            function identifier is guessed from ``f.__name__``.
        """
        if identifier is _Guess:
            # partials might not have a __name__ attribute
            identifier = getattr(f, "__name__", None)

        for kw in kwargs:
            if RE_ARGNAME.match(kw):
                # avoid collision between argument names
                raise ValueError(f"Kw argument named '{kw}' not allowed.")

        arg_id_to_arg, arg_id_to_descr = _get_arg_id_to_arg_and_arg_id_to_descr(
            args, kwargs)

        # dict_of_named_arrays = {}
        # output_id_to_name_in_program = {}

        input_id_to_name_in_program = {
            arg_id: f"_actx_in_{_ary_container_key_stringifier(arg_id)}"
            for arg_id in arg_id_to_arg}

        # Get placeholders from the ``args``, ``kwargs``.
        pl_args = [_get_f_placeholder_args(arg, iarg,
                                           input_id_to_name_in_program, actx=self)
                   for iarg, arg in enumerate(args)]

        pl_kwargs = {kw: _get_f_placeholder_args(arg, kw,
                                                 input_id_to_name_in_program,
                                                 actx=self)
                    for kw, arg in kwargs.items()}

        # Pass the placeholders
        output_template = f(*pl_args, **pl_kwargs)
        print(f"{output_template=}")

        # construct the function
        # function = FunctionDefinition(
        # frozenset(pl_arg.name for pl_arg in pl_args) | frozenset(pl_kwargs),
        # Map(returns),
        # tags=_get_default_tags() | (frozenset([FunctionIdentifier(identifier)])
        #                             if identifier
        #                             else frozenset())
        # )
        # traced_call = Call(function,
        #               (Map({pl.name: arg for pl, arg in zip(pl_args, args)})
        #                .update(Map({pl_kwargs[kw].name: arg
        #                             for kw, arg in kwargs.items()}))),
        #               result_tags=Map({name: _get_default_tags()
        #                                for name in returns}),
        #               result_axes=Map({name: _get_default_axes(ret.ndim)
        #                                for name, ret in returns.items()}),
        #               tags=_get_default_tags(),
        #               )

        # if isinstance(output, Array):
        #     return traced_call["_"]
        # elif isinstance(output, tuple):
        #    return tuple(traced_call[f"_{iarg}"] for iarg in range(len(output)))
        # elif isinstance(output, dict):
        #    return {kw: traced_call[kw] for kw in output}
        #else:
        #    raise NotImplementedError(type(output))
        return f(*args, **kwargs)

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
        import jax.numpy as jnp

        import pytato as pt
        super().__init__(compile_trace_callback=compile_trace_callback)
        self.array_types = (pt.Array, jnp.ndarray)

    @property
    def _frozen_array_types(self) -> Tuple[Type, ...]:
        import jax.numpy as jnp
        return (jnp.ndarray, )

    def _rec_map_container(
            self, func: Callable[[Array], Array], array: ArrayOrContainer,
            allowed_types: Optional[Tuple[type, ...]] = None, *,
            default_scalar: Optional[ScalarLike] = None,
            strict: bool = False) -> ArrayOrContainer:
        if allowed_types is None:
            allowed_types = self.array_types

        def _wrapper(ary):
            if isinstance(ary, allowed_types):
                return func(ary)
            elif np.isscalar(ary):
                if default_scalar is None:
                    return ary
                else:
                    return np.array(ary).dtype.type(default_scalar)
            else:
                raise TypeError(
                    f"{type(self).__name__}.{func.__name__[1:]} invoked with "
                    f"an unsupported array type: got '{type(ary).__name__}', "
                    f"but expected one of {allowed_types}")

        return rec_map_array_container(_wrapper, array)

    # {{{ ArrayContext interface

    def zeros_like(self, ary):
        from warnings import warn
        warn(f"{type(self).__name__}.zeros_like is deprecated and will stop "
            "working in 2023. Use actx.np.zeros_like instead.",
            DeprecationWarning, stacklevel=2)

        return self.np.zeros_like(ary)

    def from_numpy(self, array):
        import jax

        import pytato as pt

        def _from_numpy(ary):
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

    def freeze(self, array):
        if np.isscalar(array):
            return array

        import jax.numpy as jnp

        import pytato as pt

        from arraycontext.container.traversal import rec_keyed_map_array_container
        from arraycontext.impl.pytato.compile import _ary_container_key_stringifier

        array_as_dict: Dict[str, Union[jnp.ndarray, pt.Array]] = {}
        key_to_frozen_subary: Dict[str, jnp.ndarray] = {}
        key_to_pt_arrays: Dict[str, pt.Array] = {}

        def _record_leaf_ary_in_dict(key: Tuple[Any, ...],
                                     ary: Union[jnp.ndarray, pt.Array]) -> None:
            key_str = "_ary" + _ary_container_key_stringifier(key)
            array_as_dict[key_str] = ary

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

        def _to_frozen(key: Tuple[Any, ...], ary) -> jnp.ndarray:
            key_str = "_ary" + _ary_container_key_stringifier(key)
            return key_to_frozen_subary[key_str]

        return with_array_context(
            rec_keyed_map_array_container(_to_frozen, array),
            actx=None)

    def thaw(self, array):
        import pytato as pt

        def _thaw(ary):
            return pt.make_data_wrapper(ary)

        return with_array_context(
            self._rec_map_container(_thaw, array, self._frozen_array_types),
            actx=self)

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        from .compile import LazilyJAXCompilingFunctionCaller
        return LazilyJAXCompilingFunctionCaller(self, f)

    def tag(self, tags: ToTagSetConvertible, array):
        def _tag(ary):
            import jax.numpy as jnp
            if isinstance(ary, jnp.ndarray):
                return ary
            else:
                return ary.tagged(_preprocess_array_tags(tags))

        return self._rec_map_container(_tag, array)

    def tag_axis(self, iaxis, tags: ToTagSetConvertible, array):
        def _tag_axis(ary):
            import jax.numpy as jnp
            if isinstance(ary, jnp.ndarray):
                return ary
            else:
                return ary.with_tagged_axis(iaxis, tags)

        return self._rec_map_container(_tag_axis, array)

    # }}}

    # {{{ compilation

    def call_loopy(self, program, **kwargs):
        raise NotImplementedError(
            "Calling loopy on JAX arrays is not supported. Maybe rewrite"
            " the loopy kernel as numpy-flavored array operations using"
            " ArrayContext.np.")

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        import pytato as pt
        if arg_names is None:
            arg_names = (None,) * len(args)

        def preprocess_arg(name, arg):
            import jax.numpy as jnp
            if isinstance(arg, jnp.ndarray):
                ary = self.thaw(arg)
            elif isinstance(arg, pt.Array):
                ary = arg
            else:
                raise TypeError(
                    f"{type(self).__name__}.einsum invoked with an unsupported "
                    f"array type: got '{type(arg).__name__}', but expected one "
                    f"of {self.array_types}")

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

    def clone(self):
        return type(self)()

# }}}

# }}}

# {{{ utilities


def _ary_container_key_stringifier(keys: Tuple[Any, ...]) -> str:
    """
    Helper for :meth:`BaseLazilyCompilingFunctionCaller.__call__`. Stringifies an
    array-container's component's key. Goals of this routine:

    * No two different keys should have the same stringification
    * Stringified key must a valid identifier according to :meth:`str.isidentifier`
    * (informal) Shorter identifiers are preferred
    """
    def _rec_str(key: Any) -> str:
        if isinstance(key, (str, int)):
            return str(key)
        elif isinstance(key, tuple):
            # t in '_actx_t': stands for tuple
            return "_actx_t" + "_".join(_rec_str(k) for k in key) + "_actx_endt"
        else:
            raise NotImplementedError("Key-stringication unimplemented for "
                                      f"'{type(key).__name__}'.")

    return "_".join(_rec_str(key) for key in keys)


def _get_arg_id_to_arg_and_arg_id_to_descr(args: Tuple[Any, ...],
                                           kwargs: Mapping[str, Any]
                                           ) -> "Tuple[PMap[Tuple[Any, ...],\
                                                            Any],\
                                                       PMap[Tuple[Any, ...],\
                                                            AbstractInputDescriptor]\
                                                       ]":
    """
    Helper for :meth:`BaseLazilyCompilingFunctionCaller.__call__`. Extracts
    mappings from argument id to argument values and from argument id to
    :class:`AbstractInputDescriptor`. See
    :attr:`CompiledFunction.input_id_to_name_in_program` for argument-id's
    representation.
    """
    arg_id_to_arg: Dict[Tuple[Any, ...], Any] = {}
    arg_id_to_descr: Dict[Tuple[Any, ...], AbstractInputDescriptor] = {}

    for kw, arg in itertools.chain(enumerate(args),
                                   kwargs.items()):
        if np.isscalar(arg):
            arg_id = (kw,)
            arg_id_to_arg[arg_id] = arg
            arg_id_to_descr[arg_id] = ScalarInputDescriptor(np.dtype(type(arg)))
        elif is_array_container_type(arg.__class__):
            def id_collector(keys, ary):
                arg_id = (kw,) + keys  # noqa: B023
                arg_id_to_arg[arg_id] = ary  # noqa: B023
                arg_id_to_descr[arg_id] = LeafArrayDescriptor(  # noqa: B023
                        np.dtype(ary.dtype), ary.shape)
                return ary

            rec_keyed_map_array_container(id_collector, arg)
        elif isinstance(arg, pt.Array):
            arg_id = (kw,)
            arg_id_to_arg[arg_id] = arg
            arg_id_to_descr[arg_id] = LeafArrayDescriptor(np.dtype(arg.dtype),
                                                          arg.shape)
        else:
            raise ValueError("Argument to a compiled operator should be"
                             " either a scalar, pt.Array or an array container. Got"
                             f" '{arg}'.")

    return pmap(arg_id_to_arg), pmap(arg_id_to_descr)


def _to_input_for_compiled(ary: ArrayT, actx: PytatoPyOpenCLArrayContext):
    """
    Preprocess *ary* before turning it into a :class:`pytato.array.Placeholder`
    in :meth:`LazilyCompilingFunctionCaller.__call__`.

    Preprocessing here refers to:

    - Metadata Inference that is supplied via *actx*\'s
      :meth:`PytatoPyOpenCLArrayContext.transform_dag`.
    """
    import pyopencl.array as cla
    from arraycontext.impl.pyopencl.taggable_cl_array import (to_tagged_cl_array,
                                                              TaggableCLArray)
    if isinstance(ary, pt.Array):
        dag = pt.make_dict_of_named_arrays({"_actx_out": ary})
        # Transform the DAG to give metadata inference a chance to do its job
        return actx.transform_dag(dag)["_actx_out"].expr
    elif isinstance(ary, TaggableCLArray):
        return ary
    elif isinstance(ary, cla.Array):
        from warnings import warn
        warn("Passing pyopencl.array.Array to a compiled callable"
             " is deprecated and will stop working in 2023."
             " Use `to_tagged_cl_array` to convert the array to"
             " TaggableCLArray", DeprecationWarning, stacklevel=2)

        return to_tagged_cl_array(ary,
                                  axes=None,
                                  tags=frozenset())
    else:
        raise NotImplementedError(type(ary))


def _get_f_placeholder_args(arg, kw, arg_id_to_name, actx):
    """
    Helper for :class:`BaseLazilyCompilingFunctionCaller.__call__`. Returns the
    placeholder version of an argument to
    :attr:`BaseLazilyCompilingFunctionCaller.f`.
    """
    if np.isscalar(arg):
        name = arg_id_to_name[(kw,)]
        return pt.make_placeholder(name, (), np.dtype(type(arg)))
    elif isinstance(arg, pt.Array):
        name = arg_id_to_name[(kw,)]
        # Transform the DAG to give metadata inference a chance to do its job
        arg = _to_input_for_compiled(arg, actx)
        return pt.make_placeholder(name, arg.shape, arg.dtype,
                                   axes=arg.axes,
                                   tags=arg.tags)
    elif is_array_container_type(arg.__class__):
        def _rec_to_placeholder(keys, ary):
            name = arg_id_to_name[(kw,) + keys]
            # Transform the DAG to give metadata inference a chance to do its job
            ary = _to_input_for_compiled(ary, actx)
            return pt.make_placeholder(name,
                                       ary.shape,
                                       ary.dtype,
                                       axes=ary.axes,
                                       tags=ary.tags)

        return rec_keyed_map_array_container(_rec_to_placeholder, arg)
    else:
        raise NotImplementedError(type(arg))

# }}}

# vim: foldmethod=marker
