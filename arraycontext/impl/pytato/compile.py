"""
.. currentmodule:: arraycontext.impl.pytato.compile
.. autoclass:: PytatoCompiledOperator
.. autoclass:: PytatoExecutable
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
import numpy as np
from typing import Any, Callable, Tuple, Dict
from dataclasses import dataclass, field
from pyrsistent import pmap, PMap

import pyopencl.array as cla
import pytato as pt


class AbstractInputDescriptor:
    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError


@dataclass(frozen=True, eq=True)
class ScalarInputDescriptor(AbstractInputDescriptor):
    dtype: np.dtype


@dataclass(frozen=True, eq=True)
class ArrayContainerInputDescriptor(AbstractInputDescriptor):
    id_to_ary_descr: "PMap[Tuple[Any, ...], Tuple[np.dtype, \
                                                  Tuple[int, ...]]]"


def _ary_container_key_stringifier(keys: Tuple[Any, ...]) -> str:
    """
    Helper for :meth:`PytatoCompiledOperator.__call__`. Stringifies an
    array-container's component's key. The aim is that no-two different keys
    have the same stringification.
    """
    def _rec_str(key: Any) -> str:
        if isinstance(key, (str, int)):
            return str(key)
        elif isinstance(key, tuple):
            return "tup" + "_".join(_rec_str(k) for k in key) + "endtup"
        else:
            raise NotImplementedError

    return "_".join(_rec_str(key) for key in keys)


@dataclass
class PytatoCompiledOperator:
    """
    Records a side-effect-free callable :attr:`PytatoCompiledOperator.f`, that
    would be specialized for different input types
    :meth:`PytatoCompiledOperator.__call__` is invoked with.

    .. attribute:: f

        The callable that would be specialized into :mod:`pytato` DAGs.

    .. automethod:: __call__
    """

    actx: PytatoPyOpenCLArrayContext
    f: Callable[..., Any]
    program_cache: Dict[Tuple[AbstractInputDescriptor, ...],
                        "PytatoExecutable"] = field(default_factory=lambda: {})

    def __call__(self, *args: Any) -> Any:
        """
        Mimics :attr:`~PytatoCompiledOperator.f` being called with *args*.
        Before calling :attr:`~PytatoCompiledOperator.f`, it is compiled to a
        :mod:`pytato` DAG that would apply :attr:`~PytatoCompiledOperator.f`
        with *args* in a lazy-sense.
        """

        from arraycontext.container.traversal import (rec_keyed_map_array_container,
                                                      is_array_container)

        def to_arg_descr(arg: Any) -> AbstractInputDescriptor:
            if np.isscalar(arg):
                return ScalarInputDescriptor(np.dtype(arg))
            elif is_array_container(arg):
                id_to_ary_descr = {}

                def id_collector(keys, ary):
                    id_to_ary_descr[keys] = (np.dtype(ary.dtype),
                                             ary.shape)
                    return ary

                rec_keyed_map_array_container(id_collector, arg)
                return ArrayContainerInputDescriptor(pmap(id_to_ary_descr))
            else:
                raise ValueError("Argument to a compiled operator should be"
                                 " either a scalar or an array container. Got"
                                 f" '{arg}'.")

        arg_descrs = tuple(to_arg_descr(arg) for arg in args)

        try:
            exec_f = self.program_cache[arg_descrs]
        except KeyError:
            pass
        else:
            return exec_f(*args)

        dict_of_named_arrays = {}
        # output_naming_map: result id to name of the named array in the
        # generated pytato DAG.
        output_naming_map = {}
        # input_naming_map: argument id to placeholder name in the generated
        # pytato DAG.
        input_naming_map = {}

        def to_placeholder(arg, pos):
            if np.isscalar(arg):
                name = f"_actx_in_{pos}"
                input_naming_map[(pos, )] = name
                return pt.make_placeholder((), np.dtype(arg), name)
            elif is_array_container(arg):
                def _rec_to_placeholder(keys, ary):
                    name = (f"_actx_in_{pos}_"
                            + _ary_container_key_stringifier(keys))
                    input_naming_map[(pos,) + keys] = name
                    return pt.make_placeholder(ary.shape, ary.dtype,
                                               name)
                return rec_keyed_map_array_container(_rec_to_placeholder,
                                                     arg)
            else:
                raise NotImplementedError(type(arg))

        outputs = self.f(*[to_placeholder(arg, iarg)
                           for iarg, arg in enumerate(args)])

        if not is_array_container(outputs):
            # TODO: We could possibly just short-circuit this interface if the
            # returned type is a scalar. Not sure if it's worth it though.
            raise ValueError(f"Function '{self.f.__name__}' to be compiled did not"
                             f" return an array container, but '{outputs}' instead.")

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

        self.program_cache[arg_descrs] = PytatoExecutable(self.actx,
                                                          pytato_program,
                                                          input_naming_map,
                                                          output_naming_map,
                                                          output_template=outputs)

        return self.program_cache[arg_descrs](*args)


@dataclass
class PytatoExecutable:
    """
    A callable which is an instance of :attr:`~PytatoCompiledOperator.f`
    specialized for a particular input type fed to it.

    .. attribute:: pytato_program

    .. attribute:: input_id_to_name_in_program

        A mapping from input id to the placholder name in
        :attr:`PytatoExecutable.pytato_program`. Input id is represented as the
        position of :attr:`~PytatoCompiledOperator.f`'s argument augmented with
        the leaf array's key if the argument is an array container.

    .. attribute:: output_id_to_name_in_program

        A mapping from output id to the name of
        :class:`pytato.array.NamedArray` in
        :attr:`PytatoExecutable.pytato_program`. Output id is represented by
        the key of a leaf array in the array container
        :attr:`PytatoExecutable.output_template`.

    .. attribute:: output_template

       An instance of :class:`arraycontext.ArrayContainer` that is the return
       type of the callable.
    """

    actx: PytatoPyOpenCLArrayContext
    pytato_program: pt.target.BoundProgram
    input_id_to_name_in_program: Dict[Tuple[Any, ...], str]
    output_id_to_name_in_program: Dict[Tuple[Any, ...], str]
    output_template: ArrayContainer

    def __call__(self, *args: Any) -> ArrayContainer:
        from arraycontext.container import is_array_container
        from arraycontext.container.traversal import rec_keyed_map_array_container

        input_kwargs_to_loopy = {}

        # {{{ extract loopy arguments execute the program

        for pos, arg in enumerate(args):
            if np.isscalar(arg):
                input_kwargs_to_loopy[self.input_id_to_name_in_program[(pos,)]] = (
                    cla.to_device(self.actx.queue, np.array(arg)))
            elif is_array_container(arg):
                def _extract_lpy_kwargs(keys, ary):
                    if isinstance(ary, pt.array.DataWrapper):
                        processed_ary = ary.data
                    elif isinstance(ary, cla.Array):
                        processed_ary = ary
                    elif isinstance(ary, pt.Array):
                        processed_ary = (self.actx.freeze(ary)
                                         .with_queue(self.actx.queue))
                    else:
                        raise TypeError("Expect pytato.Array or CL-array, got "
                                f"{type(ary)}")

                    input_kwargs_to_loopy[
                        self.input_id_to_name_in_program[(pos,)
                                                         + keys]] = processed_ary
                    return ary

                rec_keyed_map_array_container(_extract_lpy_kwargs, arg)
            else:
                raise NotImplementedError(type(arg))

        # {{{ the generated program might not have depended on some of the
        # inputs => do not pass those to the loopy kernel

        input_kwargs_to_loopy = {arg_name: arg
                                 for arg_name, arg in input_kwargs_to_loopy.items()
                                 if arg_name in (self.pytato_program
                                                 .program.default_entrypoint
                                                 .arg_dict)}

        # }}}

        evt, out_dict = self.pytato_program(queue=self.actx.queue,
                                            allocator=self.actx.allocator,
                                            **input_kwargs_to_loopy)

        evt.wait()

        # }}}

        def to_output_template(keys, _):
            return self.actx.thaw(out_dict[self.output_id_to_name_in_program[keys]])

        return rec_keyed_map_array_container(to_output_template,
                                             self.output_template)
