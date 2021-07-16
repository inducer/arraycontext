"""
.. currentmodule:: arraycontext.impl.pytato.compile
.. autoclass:: LazilyCompilingFunctionCaller
.. autoclass:: CompiledFunction
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

from arraycontext.container import ArrayContainer
from arraycontext import PytatoPyOpenCLArrayContext
from arraycontext.container.traversal import (rec_keyed_map_array_container,
                                              is_array_container)

import numpy as np
from typing import Any, Callable, Tuple, Dict, Mapping
from dataclasses import dataclass, field
from pyrsistent import pmap, PMap

import pyopencl.array as cla
import pytato as pt


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
    shape: Tuple[int, ...]

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


def _get_arg_id_to_arg_and_arg_id_to_descr(args: Tuple[Any, ...]
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

    for iarg, arg in enumerate(args):
        if np.isscalar(arg):
            arg_id = (iarg,)
            arg_id_to_arg[arg_id] = arg
            arg_id_to_descr[arg_id] = ScalarInputDescriptor(np.dtype(arg))
        elif is_array_container(arg):
            def id_collector(keys, ary):
                arg_id = (iarg,) + keys
                arg_id_to_arg[arg_id] = ary
                arg_id_to_descr[arg_id] = LeafArrayDescriptor(np.dtype(ary.dtype),
                                                              ary.shape)
                return ary

            rec_keyed_map_array_container(id_collector, arg)
        else:
            raise ValueError("Argument to a compiled operator should be"
                             " either a scalar or an array container. Got"
                             f" '{arg}'.")

    return pmap(arg_id_to_arg), pmap(arg_id_to_descr)


def _get_f_placeholder_args(arg, iarg, arg_id_to_name):
    """
    Helper for :class:`LazilyCompilingFunctionCaller.__call__`. Returns the
    placeholder version of an argument to
    :attr:`LazilyCompilingFunctionCaller.f`.
    """
    if np.isscalar(arg):
        name = arg_id_to_name[(iarg,)]
        return pt.make_placeholder(name, (), np.dtype(arg))
    elif is_array_container(arg):
        def _rec_to_placeholder(keys, ary):
            name = arg_id_to_name[(iarg,) + keys]
            return pt.make_placeholder(name, ary.shape, ary.dtype)
        return rec_keyed_map_array_container(_rec_to_placeholder,
                                                arg)
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

    def __call__(self, *args: Any) -> Any:
        """
        Returns the result of :attr:`~LazilyCompilingFunctionCaller.f`'s
        function application on *args*.

        Before applying :attr:`~LazilyCompilingFunctionCaller.f`, it is compiled
        to a :mod:`pytato` DAG that would apply
        :attr:`~LazilyCompilingFunctionCaller.f` with *args* in a lazy-sense.
        The intermediary pytato DAG for *args* is memoized in *self*.
        """
        from pytato.target.loopy import BoundPyOpenCLProgram
        arg_id_to_arg, arg_id_to_descr = _get_arg_id_to_arg_and_arg_id_to_descr(args)

        try:
            compiled_f = self.program_cache[arg_id_to_descr]
        except KeyError:
            pass
        else:
            return compiled_f(arg_id_to_arg)

        dict_of_named_arrays = {}
        # output_naming_map: result id to name of the named array in the
        # generated pytato DAG.
        output_naming_map = {}
        # input_naming_map: argument id to placeholder name in the generated
        # pytato DAG.
        input_naming_map = {
            arg_id: f"_actx_in_{_ary_container_key_stringifier(arg_id)}"
            for arg_id in arg_id_to_arg}

        outputs = self.f(*[_get_f_placeholder_args(arg, iarg, input_naming_map)
                           for iarg, arg in enumerate(args)])

        if not is_array_container(outputs):
            # TODO: We could possibly just short-circuit this interface if the
            # returned type is a scalar. Not sure if it's worth it though.
            raise NotImplementedError(
                f"Function '{self.f.__name__}' to be compiled "
                "did not return an array container, but an instance of "
                f"'{outputs.__class__}' instead.")

        def _as_dict_of_named_arrays(keys, ary):
            name = "_pt_out_" + "_".join(str(key)
                                         for key in keys)
            output_naming_map[keys] = name
            dict_of_named_arrays[name] = ary
            return ary

        rec_keyed_map_array_container(_as_dict_of_named_arrays,
                                      outputs)

        pytato_program = pt.generate_loopy(dict_of_named_arrays,
                                           options={"return_dict": True},
                                           cl_device=self.actx.queue.device)
        assert isinstance(pytato_program, BoundPyOpenCLProgram)

        pytato_program = (pytato_program
                          .with_transformed_program(self
                                                    .actx
                                                    .transform_loopy_program))

        self.program_cache[arg_id_to_descr] = CompiledFunction(
                                                self.actx, pytato_program,
                                                input_naming_map, output_naming_map,
                                                output_template=outputs)

        return self.program_cache[arg_id_to_descr](arg_id_to_arg)


@dataclass
class CompiledFunction:
    """
    A callable which captures the :class:`pytato.target.BoundProgram`  resulting
    from calling :attr:`~LazilyCompilingFunctionCaller.f` with a given set of
    input types, and generating :mod:`loopy` IR from it.

    .. attribute:: pytato_program

    .. attribute:: input_id_to_name_in_program

        A mapping from input id to the placholder name in
        :attr:`CompiledFunction.pytato_program`. Input id is represented as the
        position of :attr:`~LazilyCompilingFunctionCaller.f`'s argument augmented
        with the leaf array's key if the argument is an array container.

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
        """
        :arg arg_id_to_arg: Mapping from input id to the passed argument. See
            :attr:`CompiledFunction.input_id_to_name_in_program` for input id's
            representation.
        """
        from arraycontext.container.traversal import rec_keyed_map_array_container

        input_kwargs_to_loopy = {}

        # {{{ preprocess args to get arguments (CL buffers) to be fed to the
        # loopy program

        for arg_id, arg in arg_id_to_arg.items():
            if np.isscalar(arg):
                arg = cla.to_device(self.actx.queue, np.array(arg))
            elif isinstance(arg, pt.array.DataWrapper):
                # got a Datwwrapper => simply gets its data
                arg = arg.data
            elif isinstance(arg, cla.Array):
                # got a frozen array  => do nothing
                pass
            elif isinstance(arg, pt.Array):
                # got an array expression => evaluate it
                arg = self.actx.freeze(arg).with_queue(self.actx.queue)
            else:
                raise NotImplementedError(type(arg))

            input_kwargs_to_loopy[self.input_id_to_name_in_program[arg_id]] = arg

        evt, out_dict = self.pytato_program(queue=self.actx.queue,
                                            allocator=self.actx.allocator,
                                            **input_kwargs_to_loopy)
        # FIXME Kernels (for now) allocate tons of memory in temporaries. If we
        # race too far ahead with enqueuing, there is a distinct risk of
        # running out of memory. This mitigates that risk a bit, for now.
        evt.wait()

        # }}}

        def to_output_template(keys, _):
            return self.actx.thaw(out_dict[self.output_id_to_name_in_program[keys]])

        return rec_keyed_map_array_container(to_output_template,
                                             self.output_template)
