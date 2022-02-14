"""
.. currentmodule:: arraycontext.impl.pytato.compile
.. autoclass:: LazilyCompilingFunctionCaller
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

from arraycontext.container import ArrayContainer, is_array_container_type
from arraycontext import PytatoPyOpenCLArrayContext
from arraycontext.container.traversal import rec_keyed_map_array_container

import abc
import numpy as np
from typing import Any, Callable, Tuple, Dict, Mapping
from dataclasses import dataclass, field
from pyrsistent import pmap, PMap

import pyopencl.array as cla
import pytato as pt
import itertools
from pytools.tag import Tag

from pytools import ProcessLogger

import logging
logger = logging.getLogger(__name__)


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
    Used internally in :class:`LazilyCompilingFunctionCaller` to characterize
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


def _ary_container_key_stringifier(keys: Tuple[Any, ...]) -> str:
    """
    Helper for :meth:`LazilyCompilingFunctionCaller.__call__`. Stringifies an
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
    Helper for :meth:`LazilyCompilingFunctionCaller.__call__`. Extracts
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
                arg_id = (kw,) + keys
                arg_id_to_arg[arg_id] = ary
                arg_id_to_descr[arg_id] = LeafArrayDescriptor(np.dtype(ary.dtype),
                                                              ary.shape)
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


def _get_f_placeholder_args(arg, kw, arg_id_to_name):
    """
    Helper for :class:`LazilyCompilingFunctionCaller.__call__`. Returns the
    placeholder version of an argument to
    :attr:`LazilyCompilingFunctionCaller.f`.
    """
    if np.isscalar(arg):
        name = arg_id_to_name[(kw,)]
        return pt.make_placeholder(name, (), np.dtype(type(arg)))
    elif isinstance(arg, pt.Array):
        name = arg_id_to_name[(kw,)]
        return pt.make_placeholder(name, arg.shape, arg.dtype)
    elif is_array_container_type(arg.__class__):
        def _rec_to_placeholder(keys, ary):
            name = arg_id_to_name[(kw,) + keys]
            return pt.make_placeholder(name, ary.shape, ary.dtype)

        return rec_keyed_map_array_container(_rec_to_placeholder, arg)
    else:
        raise NotImplementedError(type(arg))


@dataclass
class LazilyCompilingFunctionCaller:
    """
    Records a side-effect-free callable
    :attr:`LazilyCompilingFunctionCaller.f` that can be specialized for the
    input types with which :meth:`LazilyCompilingFunctionCaller.__call__` is
    invoked.

    .. attribute:: f

        The callable that will be called to obtain :mod:`pytato` DAGs.

    .. automethod:: __call__
    """

    actx: PytatoPyOpenCLArrayContext
    f: Callable[..., Any]
    program_cache: Dict["PMap[Tuple[Any, ...], AbstractInputDescriptor]",
                        "CompiledFunction"] = field(default_factory=lambda: {})

    def _dag_to_transformed_loopy_prg(self, dict_of_named_arrays):
        from pytato.target.loopy import BoundPyOpenCLProgram

        import loopy as lp

        with ProcessLogger(logger, "transform_dag"):
            pt_dict_of_named_arrays = self.actx.transform_dag(dict_of_named_arrays)

        with ProcessLogger(logger, "generate_loopy"):
            pytato_program = pt.generate_loopy(pt_dict_of_named_arrays,
                                               options=lp.Options(
                                                   return_dict=True,
                                                   no_numpy=True),
                                               cl_device=self.actx.queue.device)
            assert isinstance(pytato_program, BoundPyOpenCLProgram)

        with ProcessLogger(logger, "transform_loopy_program"):

            pytato_program = (pytato_program
                              .with_transformed_program(
                                  lambda x: x.with_kernel(
                                      x.default_entrypoint
                                      .tagged(FromArrayContextCompile()))))

            pytato_program = (pytato_program
                              .with_transformed_program(self
                                                        .actx
                                                        .transform_loopy_program))

        return pytato_program

    def _dag_to_compiled_func(self, ary_or_dict_of_named_arrays,
            input_id_to_name_in_program, output_id_to_name_in_program,
            output_template):
        if isinstance(ary_or_dict_of_named_arrays, pt.Array):
            output_id = "_pt_out"
            dict_of_named_arrays = pt.make_dict_of_named_arrays(
                {output_id: ary_or_dict_of_named_arrays})
            pytato_program = self._dag_to_transformed_loopy_prg(dict_of_named_arrays)
            return CompiledFunctionReturningArray(
                self.actx, pytato_program,
                input_id_to_name_in_program=input_id_to_name_in_program,
                output_name_in_program=output_id)
        elif isinstance(ary_or_dict_of_named_arrays, pt.DictOfNamedArrays):
            pytato_program = self._dag_to_transformed_loopy_prg(
                ary_or_dict_of_named_arrays)
            return CompiledFunctionReturningArrayContainer(
                    self.actx, pytato_program,
                    input_id_to_name_in_program=input_id_to_name_in_program,
                    output_id_to_name_in_program=output_id_to_name_in_program,
                    output_template=output_template)
        else:
            raise NotImplementedError(type(ary_or_dict_of_named_arrays))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Returns the result of :attr:`~LazilyCompilingFunctionCaller.f`'s
        function application on *args*.

        Before applying :attr:`~LazilyCompilingFunctionCaller.f`, it is compiled
        to a :mod:`pytato` DAG that would apply
        :attr:`~LazilyCompilingFunctionCaller.f` with *args* in a lazy-sense.
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
                *[_get_f_placeholder_args(arg, iarg, input_id_to_name_in_program)
                    for iarg, arg in enumerate(args)],
                **{kw: _get_f_placeholder_args(arg, kw, input_id_to_name_in_program)
                    for kw, arg in kwargs.items()})

        if (not (is_array_container_type(output_template.__class__)
                 or isinstance(output_template, pt.Array))):
            # TODO: We could possibly just short-circuit this interface if the
            # returned type is a scalar. Not sure if it's worth it though.
            raise NotImplementedError(
                f"Function '{self.f.__name__}' to be compiled "
                "did not return an array container or pt.Array,"
                f" but an instance of '{output_template.__class__}' instead.")

        def _as_dict_of_named_arrays(keys, ary):
            name = "_pt_out_" + "_".join(str(key)
                                         for key in keys)
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


def _args_to_cl_buffers(actx, input_id_to_name_in_program, arg_id_to_arg):
    input_kwargs_for_loopy = {}

    for arg_id, arg in arg_id_to_arg.items():
        if np.isscalar(arg):
            arg = cla.to_device(actx.queue, np.array(arg))
        elif isinstance(arg, pt.array.DataWrapper):
            # got a Datwwrapper => simply gets its data
            arg = arg.data
        elif isinstance(arg, cla.Array):
            # got a frozen array  => do nothing
            pass
        elif isinstance(arg, pt.Array):
            # got an array expression => evaluate it
            arg = actx.freeze(arg).with_queue(actx.queue)
        else:
            raise NotImplementedError(type(arg))

        input_kwargs_for_loopy[input_id_to_name_in_program[arg_id]] = arg

    return input_kwargs_for_loopy


class CompiledFunction(abc.ABC):
    """
    A callable which captures the :class:`pytato.target.BoundProgram`  resulting
    from calling :attr:`~LazilyCompilingFunctionCaller.f` with a given set of
    input types, and generating :mod:`loopy` IR from it.

    .. attribute:: pytato_program

    .. attribute:: input_id_to_name_in_program

        A mapping from input id to the placeholder name in
        :attr:`CompiledFunction.pytato_program`. Input id is represented as the
        position of :attr:`~LazilyCompilingFunctionCaller.f`'s argument augmented
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


@dataclass(frozen=True)
class CompiledFunctionReturningArrayContainer(CompiledFunction):
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
    output_template: ArrayContainer

    def __call__(self, arg_id_to_arg) -> ArrayContainer:
        input_kwargs_for_loopy = _args_to_cl_buffers(
                self.actx, self.input_id_to_name_in_program, arg_id_to_arg)

        evt, out_dict = self.pytato_program(queue=self.actx.queue,
                                            allocator=self.actx.allocator,
                                            **input_kwargs_for_loopy)

        # FIXME Kernels (for now) allocate tons of memory in temporaries. If we
        # race too far ahead with enqueuing, there is a distinct risk of
        # running out of memory. This mitigates that risk a bit, for now.
        evt.wait()

        def to_output_template(keys, _):
            return self.actx.thaw(out_dict[self.output_id_to_name_in_program[keys]])

        return rec_keyed_map_array_container(to_output_template,
                                             self.output_template)


@dataclass(frozen=True)
class CompiledFunctionReturningArray(CompiledFunction):
    """
    .. attribute:: output_name_in_program

        Name of the output array in the program.
    """
    actx: PytatoPyOpenCLArrayContext
    pytato_program: pt.target.BoundProgram
    input_id_to_name_in_program: Mapping[Tuple[Any, ...], str]
    output_name: str

    def __call__(self, arg_id_to_arg) -> ArrayContainer:
        input_kwargs_for_loopy = _args_to_cl_buffers(
                self.actx, self.input_id_to_name_in_program, arg_id_to_arg)

        evt, out_dict = self.pytato_program(queue=self.actx.queue,
                                            allocator=self.actx.allocator,
                                            **input_kwargs_for_loopy)

        # FIXME Kernels (for now) allocate tons of memory in temporaries. If we
        # race too far ahead with enqueuing, there is a distinct risk of
        # running out of memory. This mitigates that risk a bit, for now.
        evt.wait()

        return self.actx.thaw(out_dict[self.output_name])
