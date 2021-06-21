"""
.. currentmodule:: arraycontext
.. autoclass:: PytatoPyOpenCLArrayContext
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
from typing import Any, Callable, Tuple, Union, Mapping
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
    id_to_ary_descr: "PMap[Tuple[Union[str, int], ...], Tuple[np.dtype, \
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
    actx: ArrayContext
    f: Callable[[Any], Any]
    program_cache: Mapping[Tuple[AbstractInputDescriptor],
                           "PytatoExecutable"] = field(default_factory=lambda: {})

    def __call__(self, *args):

        from arraycontext.container.traversal import (rec_keyed_map_array_container,
                                                      is_array_container)

        def to_arg_descr(arg):
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


class PytatoExecutable:
    def __init__(self, actx, pytato_program, input_id_to_name_in_program,
                 output_id_to_name_in_program, output_template):
        self.actx = actx
        self.pytato_program = pytato_program
        self.input_id_to_name_in_program = input_id_to_name_in_program
        self.output_id_to_name_in_program = output_id_to_name_in_program
        self.output_template = output_template

    def __call__(self, *args):
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
