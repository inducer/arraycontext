"""
.. autoclass:: BaseLazilyCompilingFunctionCaller
.. autoclass:: LazilyPyOpenCLCompilingFunctionCaller
.. autoclass:: LazilyJAXCompilingFunctionCaller
.. autoclass:: CompiledFunction
.. autoclass:: FromArrayContextCompile
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
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Mapping, Tuple, Type

import numpy as np
from immutabledict import immutabledict

import pytato as pt
from pytools import ProcessLogger
from pytools.tag import Tag

from arraycontext.container import ArrayContainer, is_array_container_type
from arraycontext.container.traversal import rec_keyed_map_array_container
from arraycontext.context import ArrayT
from arraycontext.impl.pytato import (
    PytatoJAXArrayContext, PytatoPyOpenCLArrayContext, _BasePytatoArrayContext)


logger = logging.getLogger(__name__)


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
                                           ) -> \
            Tuple[Mapping[Tuple[Any, ...], Any],
                  Mapping[Tuple[Any, ...], AbstractInputDescriptor]]:
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

    return immutabledict(arg_id_to_arg), immutabledict(arg_id_to_descr)


def _to_input_for_compiled(ary: ArrayT, actx: PytatoPyOpenCLArrayContext):
    """
    Preprocess *ary* before turning it into a :class:`pytato.array.Placeholder`
    in :meth:`LazilyCompilingFunctionCaller.__call__`.

    Preprocessing here refers to:

    - Metadata Inference that is supplied via *actx*\'s
      :meth:`PytatoPyOpenCLArrayContext.transform_dag`.
    """
    import pyopencl.array as cla

    from arraycontext.impl.pyopencl.taggable_cl_array import (
        TaggableCLArray, to_tagged_cl_array)
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


# {{{ BaseLazilyCompilingFunctionCaller

@dataclass
class BaseLazilyCompilingFunctionCaller:
    """
    Records a side-effect-free callable :attr:`f` that can be specialized for
    the input types with which :meth:`__call__` is invoked.

    .. attribute:: f

        The callable that will be called to obtain :mod:`pytato` DAGs.

    .. automethod:: __call__
    """

    actx: _BasePytatoArrayContext
    f: Callable[..., Any]
    program_cache: Dict[Mapping[Tuple[Any, ...], AbstractInputDescriptor],
                        "CompiledFunction"] = field(default_factory=lambda: {})

    # {{{ abstract interface

    def _dag_to_transformed_pytato_prg(self, dict_of_named_arrays, *, prg_id=None):
        raise NotImplementedError

    @property
    def compiled_function_returning_array_container_class(
            self) -> Type["CompiledFunction"]:
        raise NotImplementedError

    @property
    def compiled_function_returning_array_class(self) -> Type["CompiledFunction"]:
        raise NotImplementedError

    # }}}

    def _dag_to_compiled_func(self, ary_or_dict_of_named_arrays,
            input_id_to_name_in_program, output_id_to_name_in_program,
            output_template):
        if isinstance(ary_or_dict_of_named_arrays, pt.Array):
            output_id = "_pt_out"
            dict_of_named_arrays = pt.make_dict_of_named_arrays(
                {output_id: ary_or_dict_of_named_arrays})
            pytato_program, name_in_program_to_tags, name_in_program_to_axes = (
                self._dag_to_transformed_pytato_prg(dict_of_named_arrays,
                    prg_id=self.f))
            return self.compiled_function_returning_array_class(
                self.actx, pytato_program,
                input_id_to_name_in_program=input_id_to_name_in_program,
                output_tags=name_in_program_to_tags[output_id],
                output_axes=name_in_program_to_axes[output_id],
                output_name=output_id)
        elif isinstance(ary_or_dict_of_named_arrays, pt.DictOfNamedArrays):
            pytato_program, name_in_program_to_tags, name_in_program_to_axes = (
                self._dag_to_transformed_pytato_prg(ary_or_dict_of_named_arrays,
                    prg_id=self.f))
            return self.compiled_function_returning_array_container_class(
                    self.actx, pytato_program,
                    input_id_to_name_in_program=input_id_to_name_in_program,
                    output_id_to_name_in_program=output_id_to_name_in_program,
                    name_in_program_to_tags=name_in_program_to_tags,
                    name_in_program_to_axes=name_in_program_to_axes,
                    output_template=output_template)
        else:
            raise NotImplementedError(type(ary_or_dict_of_named_arrays))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Returns the result of :attr:`~BaseLazilyCompilingFunctionCaller.f`'s
        function application on *args*.

        Before applying :attr:`~BaseLazilyCompilingFunctionCaller.f`, it is compiled
        to a :mod:`pytato` DAG that would apply
        :attr:`~BaseLazilyCompilingFunctionCaller.f` with *args* in a lazy-sense.
        The intermediary pytato DAG for *args* is memoized in *self*.
        """
        arg_id_to_arg, arg_id_to_descr = _get_arg_id_to_arg_and_arg_id_to_descr(
            args, kwargs)

        try:
            compiled_f = self.program_cache[arg_id_to_descr]
        except KeyError:
            pass
        else:
            return compiled_f(arg_id_to_arg)

        dict_of_named_arrays = {}
        output_id_to_name_in_program = {}
        input_id_to_name_in_program = {
            arg_id: f"_actx_in_{_ary_container_key_stringifier(arg_id)}"
            for arg_id in arg_id_to_arg}

        output_template = self.f(
                *[_get_f_placeholder_args(arg, iarg,
                                          input_id_to_name_in_program, self.actx)
                    for iarg, arg in enumerate(args)],
                **{kw: _get_f_placeholder_args(arg, kw,
                                               input_id_to_name_in_program,
                                               self.actx)
                    for kw, arg in kwargs.items()})

        self.actx._compile_trace_callback(self.f, "post_trace", output_template)

        if (not (is_array_container_type(output_template.__class__)
                 or isinstance(output_template, pt.Array))):
            # TODO: We could possibly just short-circuit this interface if the
            # returned type is a scalar. Not sure if it's worth it though.
            raise NotImplementedError(
                f"Function '{self.f.__name__}' to be compiled "
                "did not return an array container or pt.Array,"
                f" but an instance of '{output_template.__class__}' instead.")

        def _as_dict_of_named_arrays(keys, ary):
            name = "_pt_out_" + _ary_container_key_stringifier(keys)
            output_id_to_name_in_program[keys] = name
            dict_of_named_arrays[name] = ary
            return ary

        rec_keyed_map_array_container(_as_dict_of_named_arrays,
                                      output_template)

        compiled_func = self._dag_to_compiled_func(
                pt.make_dict_of_named_arrays(dict_of_named_arrays),
                input_id_to_name_in_program=input_id_to_name_in_program,
                output_id_to_name_in_program=output_id_to_name_in_program,
                output_template=output_template)

        self.program_cache[arg_id_to_descr] = compiled_func
        return compiled_func(arg_id_to_arg)

# }}}


# {{{ LazilyPyOpenCLCompilingFunctionCaller

class LazilyPyOpenCLCompilingFunctionCaller(BaseLazilyCompilingFunctionCaller):
    actx: PytatoPyOpenCLArrayContext

    @property
    def compiled_function_returning_array_container_class(
            self) -> Type["CompiledFunction"]:
        return CompiledPyOpenCLFunctionReturningArrayContainer

    @property
    def compiled_function_returning_array_class(self) -> Type["CompiledFunction"]:
        return CompiledPyOpenCLFunctionReturningArray

    def _dag_to_transformed_pytato_prg(self, dict_of_named_arrays, *, prg_id=None):
        if prg_id is None:
            prg_id = self.f

        from pytato.target.loopy import BoundPyOpenCLExecutable

        self.actx._compile_trace_callback(
                prg_id, "pre_transform_dag", dict_of_named_arrays)

        with ProcessLogger(logger, f"transform_dag for '{prg_id}'"):
            pt_dict_of_named_arrays = self.actx.transform_dag(dict_of_named_arrays)

        self.actx._compile_trace_callback(
                prg_id, "post_transform_dag", pt_dict_of_named_arrays)

        name_in_program_to_tags = {
            name: out.tags
            for name, out in pt_dict_of_named_arrays._data.items()}
        name_in_program_to_axes = {
            name: out.axes
            for name, out in pt_dict_of_named_arrays._data.items()}

        self.actx._compile_trace_callback(
                prg_id, "pre_generate_loopy", pt_dict_of_named_arrays)

        with ProcessLogger(logger, f"generate_loopy for '{prg_id}'"):
            from arraycontext.loopy import _DEFAULT_LOOPY_OPTIONS
            opts = _DEFAULT_LOOPY_OPTIONS
            assert opts.return_dict

            pytato_program = pt.generate_loopy(
                    pt_dict_of_named_arrays,
                    options=opts,
                    function_name=_prg_id_to_kernel_name(prg_id),
                    target=self.actx.get_target(),
                    ).bind_to_context(self.actx.context)  # pylint: disable=no-member
            assert isinstance(pytato_program, BoundPyOpenCLExecutable)

        self.actx._compile_trace_callback(
                prg_id, "post_generate_loopy", pytato_program)

        self.actx._compile_trace_callback(
                prg_id, "pre_transform_loopy_program", pytato_program)

        with ProcessLogger(logger, f"transform_loopy_program for '{prg_id}'"):

            pytato_program = (pytato_program
                              .with_transformed_translation_unit(
                                  lambda x: x.with_kernel(
                                      x.default_entrypoint
                                      .tagged(FromArrayContextCompile()))))

            pytato_program = (pytato_program
                              .with_transformed_translation_unit(
                                  self.actx.transform_loopy_program))

        self.actx._compile_trace_callback(
                prg_id, "post_transform_loopy_program", pytato_program)

        self.actx._compile_trace_callback(
                prg_id, "final", pytato_program)

        return pytato_program, name_in_program_to_tags, name_in_program_to_axes

# }}}


# {{{ preserve back compat

class LazilyCompilingFunctionCaller(LazilyPyOpenCLCompilingFunctionCaller):
    def __new__(cls, *args, **kwargs):
        from warnings import warn
        warn("LazilyCompilingFunctionCaller has been renamed to"
             " LazilyPyOpenCLCompilingFunctionCaller. This will be"
             " an error in 2023.", DeprecationWarning, stacklevel=2)
        return super(LazilyCompilingFunctionCaller, cls).__new__(cls)

    def _dag_to_transformed_loopy_prg(self, dict_of_named_arrays):
        from warnings import warn
        warn("_dag_to_transformed_loopy_prg has been renamed to"
             " _dag_to_transformed_pytato_prg. This will be"
             " an error in 2023.", DeprecationWarning, stacklevel=2)
        return super()._dag_to_transformed_pytato_prg(dict_of_named_arrays)

# }}}


# {{{ LazilyJAXCompilingFunctionCaller

class LazilyJAXCompilingFunctionCaller(BaseLazilyCompilingFunctionCaller):
    @property
    def compiled_function_returning_array_container_class(
            self) -> Type["CompiledFunction"]:
        return CompiledJAXFunctionReturningArrayContainer

    @property
    def compiled_function_returning_array_class(self) -> Type["CompiledFunction"]:
        return CompiledJAXFunctionReturningArray

    def _dag_to_transformed_pytato_prg(self, dict_of_named_arrays, *, prg_id=None):
        if prg_id is None:
            prg_id = self.f

        self.actx._compile_trace_callback(
                prg_id, "pre_transform_dag", dict_of_named_arrays)

        with ProcessLogger(logger, "transform_dag for '{prg_id}'"):
            pt_dict_of_named_arrays = self.actx.transform_dag(dict_of_named_arrays)

        self.actx._compile_trace_callback(
                prg_id, "post_transform_dag", pt_dict_of_named_arrays)

        name_in_program_to_tags = {
            name: out.tags
            for name, out in pt_dict_of_named_arrays._data.items()}
        name_in_program_to_axes = {
            name: out.axes
            for name, out in pt_dict_of_named_arrays._data.items()}

        self.actx._compile_trace_callback(
                prg_id, "pre_generate_jax", pt_dict_of_named_arrays)

        with ProcessLogger(logger, f"generate_jax for '{prg_id}'"):
            pytato_program = pt.generate_jax(
                    pt_dict_of_named_arrays,
                    jit=True,
                    function_name=_prg_id_to_kernel_name(prg_id))

        self.actx._compile_trace_callback(
                prg_id, "post_generate_jax", pytato_program)

        return pytato_program, name_in_program_to_tags, name_in_program_to_axes


def _args_to_device_buffers(actx, input_id_to_name_in_program, arg_id_to_arg):
    input_kwargs_for_loopy = {}

    for arg_id, arg in arg_id_to_arg.items():
        if np.isscalar(arg):
            if isinstance(actx, PytatoPyOpenCLArrayContext):
                import pyopencl.array as cla
                arg = cla.to_device(actx.queue, np.array(arg),
                        allocator=actx.allocator)
            elif isinstance(actx, PytatoJAXArrayContext):
                import jax
                arg = jax.device_put(arg)
            else:
                raise NotImplementedError(type(actx))

        elif isinstance(arg, pt.array.DataWrapper):
            # got a Datawrapper => simply gets its data
            arg = arg.data
        elif isinstance(arg, actx._frozen_array_types):
            # got a frozen array  => do nothing
            pass
        elif isinstance(arg, pt.Array):
            # got an array expression => evaluate it
            from warnings import warn
            warn(f"Argument array '{arg_id}' to a compiled function is "
                    "unevaluated. Evaluating just-in-time, at "
                    "considerable expense. This is deprecated and will stop "
                    "working in 2023. To avoid this warning, force evaluation "
                    "of all arguments via freeze/thaw.",
                    DeprecationWarning, stacklevel=4)

            arg = actx.freeze(arg)
        else:
            raise NotImplementedError(type(arg))

        input_kwargs_for_loopy[input_id_to_name_in_program[arg_id]] = arg

    return input_kwargs_for_loopy


def _args_to_cl_buffers(actx, input_id_to_name_in_program, arg_id_to_arg):
    from warnings import warn
    warn("_args_to_cl_buffer has been renamed to"
         " _args_to_device_buffers. This will be"
         " an error in 2023.", DeprecationWarning, stacklevel=2)
    return _args_to_device_buffers(actx, input_id_to_name_in_program,
                                   arg_id_to_arg)

# }}}


# {{{ compiled function

class CompiledFunction(abc.ABC):
    """
    A callable which captures the :class:`pytato.target.BoundProgram`  resulting
    from calling :attr:`~BaseLazilyCompilingFunctionCaller.f` with a given set of
    input types, and generating :mod:`loopy` IR from it.

    .. attribute:: pytato_program

    .. attribute:: input_id_to_name_in_program

        A mapping from input id to the placeholder name in
        :attr:`CompiledFunction.pytato_program`. Input id is represented as the
        position of :attr:`~BaseLazilyCompilingFunctionCaller.f`'s argument augmented
        with the leaf array's key if the argument is an array container.


    .. automethod:: __call__
    """

    @abc.abstractmethod
    def __call__(self, arg_id_to_arg) -> Any:
        """
        :arg arg_id_to_arg: Mapping from input id to the passed argument. See
            :attr:`CompiledFunction.input_id_to_name_in_program` for input id's
            representation.
        """
        pass

# }}}


# {{{ compiled pyopencl function

@dataclass(frozen=True)
class CompiledPyOpenCLFunctionReturningArrayContainer(CompiledFunction):
    """
    .. attribute:: output_id_to_name_in_program

        A mapping from output id to the name of
        :class:`pytato.array.NamedArray` in
        :attr:`CompiledFunction.pytato_program`. Output id is represented by
        the key of a leaf array in the array container
        :attr:`CompiledFunction.output_template`.

    .. attribute:: output_template

       An instance of :class:`arraycontext.ArrayContainer` that is the return
       type of the callable.
    """
    actx: PytatoPyOpenCLArrayContext
    pytato_program: pt.target.BoundProgram
    input_id_to_name_in_program: Mapping[Tuple[Any, ...], str]
    output_id_to_name_in_program: Mapping[Tuple[Any, ...], str]
    name_in_program_to_tags: Mapping[str, FrozenSet[Tag]]
    name_in_program_to_axes: Mapping[str, Tuple[pt.Axis, ...]]
    output_template: ArrayContainer

    def __call__(self, arg_id_to_arg) -> ArrayContainer:
        from .utils import get_cl_axes_from_pt_axes
        from arraycontext.impl.pyopencl.taggable_cl_array import to_tagged_cl_array

        input_kwargs_for_loopy = _args_to_device_buffers(
                self.actx, self.input_id_to_name_in_program, arg_id_to_arg)

        evt, out_dict = self.pytato_program(queue=self.actx.queue,
                                            allocator=self.actx.allocator,
                                            **input_kwargs_for_loopy)

        # FIXME Kernels (for now) allocate tons of memory in temporaries. If we
        # race too far ahead with enqueuing, there is a distinct risk of
        # running out of memory. This mitigates that risk a bit, for now.
        evt.wait()

        def to_output_template(keys, _):
            name_in_program = self.output_id_to_name_in_program[keys]
            return self.actx.thaw(to_tagged_cl_array(
                out_dict[name_in_program],
                axes=get_cl_axes_from_pt_axes(
                    self.name_in_program_to_axes[name_in_program]),
                tags=self.name_in_program_to_tags[name_in_program]))

        return rec_keyed_map_array_container(to_output_template,
                                             self.output_template)


@dataclass(frozen=True)
class CompiledPyOpenCLFunctionReturningArray(CompiledFunction):
    """
    .. attribute:: output_name_in_program

        Name of the output array in the program.
    """
    actx: PytatoPyOpenCLArrayContext
    pytato_program: pt.target.BoundProgram
    input_id_to_name_in_program: Mapping[Tuple[Any, ...], str]
    output_tags: FrozenSet[Tag]
    output_axes: Tuple[pt.Axis, ...]
    output_name: str

    def __call__(self, arg_id_to_arg) -> ArrayContainer:
        from .utils import get_cl_axes_from_pt_axes
        from arraycontext.impl.pyopencl.taggable_cl_array import to_tagged_cl_array

        input_kwargs_for_loopy = _args_to_device_buffers(
                self.actx, self.input_id_to_name_in_program, arg_id_to_arg)

        evt, out_dict = self.pytato_program(queue=self.actx.queue,
                                            allocator=self.actx.allocator,
                                            **input_kwargs_for_loopy)

        # FIXME Kernels (for now) allocate tons of memory in temporaries. If we
        # race too far ahead with enqueuing, there is a distinct risk of
        # running out of memory. This mitigates that risk a bit, for now.
        evt.wait()

        return self.actx.thaw(to_tagged_cl_array(out_dict[self.output_name],
                                                 axes=get_cl_axes_from_pt_axes(
                                                     self.output_axes),
                                                 tags=self.output_tags))

# }}}


# {{{ compiled jax function

@dataclass(frozen=True)
class CompiledJAXFunctionReturningArrayContainer(CompiledFunction):
    """
    .. attribute:: output_id_to_name_in_program

        A mapping from output id to the name of
        :class:`pytato.array.NamedArray` in
        :attr:`CompiledFunction.pytato_program`. Output id is represented by
        the key of a leaf array in the array container
        :attr:`CompiledFunction.output_template`.

    .. attribute:: output_template

       An instance of :class:`arraycontext.ArrayContainer` that is the return
       type of the callable.
    """
    actx: PytatoJAXArrayContext
    pytato_program: pt.target.BoundProgram
    input_id_to_name_in_program: Mapping[Tuple[Any, ...], str]
    output_id_to_name_in_program: Mapping[Tuple[Any, ...], str]
    name_in_program_to_tags: Mapping[str, FrozenSet[Tag]]
    name_in_program_to_axes: Mapping[str, Tuple[pt.Axis, ...]]
    output_template: ArrayContainer

    def __call__(self, arg_id_to_arg) -> ArrayContainer:
        input_kwargs_for_loopy = _args_to_device_buffers(
                self.actx, self.input_id_to_name_in_program, arg_id_to_arg)

        out_dict = self.pytato_program(**input_kwargs_for_loopy)

        def to_output_template(keys, _):
            return self.actx.thaw(
                out_dict[self.output_id_to_name_in_program[keys]]
                .block_until_ready()
            )

        return rec_keyed_map_array_container(to_output_template,
                                             self.output_template)


@dataclass(frozen=True)
class CompiledJAXFunctionReturningArray(CompiledFunction):
    """
    .. attribute:: output_name_in_program

        Name of the output array in the program.
    """
    actx: PytatoJAXArrayContext
    pytato_program: pt.target.BoundProgram
    input_id_to_name_in_program: Mapping[Tuple[Any, ...], str]
    output_tags: FrozenSet[Tag]
    output_axes: Tuple[pt.Axis, ...]
    output_name: str

    def __call__(self, arg_id_to_arg) -> ArrayContainer:
        input_kwargs_for_loopy = _args_to_device_buffers(
                self.actx, self.input_id_to_name_in_program, arg_id_to_arg)

        evt, out_dict = self.pytato_program(**input_kwargs_for_loopy)

        return self.actx.thaw(out_dict[self.output_name])

# }}}

# vim: foldmethod=marker
