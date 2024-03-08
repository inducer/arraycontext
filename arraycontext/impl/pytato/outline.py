"""
.. autoclass:: OutlinedCall
"""
__copyright__ = """
Copyright (C) 2023 Kaushik Kulkarni
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

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Mapping, Tuple

import numpy as np
from immutables import Map

import pytato as pt
from pytools.tag import Tag

from arraycontext.container import is_array_container_type
from arraycontext.container.traversal import rec_keyed_map_array_container
from arraycontext.context import ArrayOrContainer
from arraycontext.impl.pytato import _BasePytatoArrayContext


def _get_arg_id_to_arg(args: Tuple[Any, ...],
                       kwargs: Mapping[str, Any]
                       ) -> Map[Tuple[Any, ...], Any]:
    """
    Helper for :meth:`OulinedCall.__call__`. Extracts mappings from argument id
    to argument values. See
    :attr:`CompiledFunction.input_id_to_name_in_function` for argument-id's
    representation.
    """
    arg_id_to_arg: Dict[Tuple[Any, ...], Any] = {}

    for kw, arg in itertools.chain(enumerate(args),
                                   kwargs.items()):
        if np.isscalar(arg):
            # do not make scalars as placeholders since we inline them.
            pass
        elif is_array_container_type(arg.__class__):
            def id_collector(keys, ary):
                arg_id = (kw,) + keys  # noqa: B023
                arg_id_to_arg[arg_id] = ary  # noqa: B023
                return ary

            rec_keyed_map_array_container(id_collector, arg)
        elif isinstance(arg, pt.Array):
            arg_id = (kw,)
            arg_id_to_arg[arg_id] = arg
        else:
            raise ValueError("Argument to a compiled operator should be"
                             " either a scalar, pt.Array or an array container. Got"
                             f" '{arg}'.")

    return Map(arg_id_to_arg)


def _get_placeholder_replacement(arg, kw, arg_id_to_name):
    """
    Helper for :class:`OutlinedCall.__call__`. Returns the placeholder version
    of an argument to :attr:`OutlinedCall.f`.
    """
    if np.isscalar(arg):
        return arg
    elif isinstance(arg, pt.Array):
        name = arg_id_to_name[(kw,)]
        return pt.make_placeholder(name, arg.shape, arg.dtype)
    elif is_array_container_type(arg.__class__):
        def _rec_to_placeholder(keys, ary):
            name = arg_id_to_name[(kw,) + keys]
            return pt.make_placeholder(name,
                                       ary.shape,
                                       ary.dtype)

        return rec_keyed_map_array_container(_rec_to_placeholder, arg)
    else:
        raise NotImplementedError(type(arg))


def _get_input_arg_id_str(arg_id: Tuple[Any, ...]) -> str:
    from arraycontext.impl.pytato.utils import _ary_container_key_stringifier
    return f"_actx_in_{_ary_container_key_stringifier(arg_id)}"


def _get_output_arg_id_str(arg_id: Tuple[Any, ...]) -> str:
    from arraycontext.impl.pytato.utils import _ary_container_key_stringifier
    return f"_actx_out_{_ary_container_key_stringifier(arg_id)}"


@dataclass(frozen=True)
class OutlinedCall:
    actx: _BasePytatoArrayContext
    f: Callable[..., Any]
    tags: FrozenSet[Tag]

    def __call__(self, *args: Any, **kwargs: Any) -> ArrayOrContainer:
        arg_id_to_arg = _get_arg_id_to_arg(args, kwargs)
        input_id_to_name_in_function = {arg_id: _get_input_arg_id_str(arg_id)
                                       for arg_id in arg_id_to_arg}

        pl_args = [_get_placeholder_replacement(arg, iarg,
                                                input_id_to_name_in_function)
                   for iarg, arg in enumerate(args)]
        pl_kwargs = {kw: _get_placeholder_replacement(arg, kw,
                                                      input_id_to_name_in_function)
                     for kw, arg in kwargs.items()}

        output = self.f(*pl_args, **pl_kwargs)

        if isinstance(output, pt.Array):
            returns = {"_": output}
            ret_type = pt.function.ReturnType.ARRAY
        elif is_array_container_type(output.__class__):
            returns = {}

            def _unpack_container(key, ary):
                key = _get_output_arg_id_str(key)
                returns[key] = ary
                return ary

            rec_keyed_map_array_container(_unpack_container, output)
            ret_type = pt.function.ReturnType.DICT_OF_ARRAYS
        else:
            raise NotImplementedError(type(output))

        # pylint-disable-reason: pylint has a hard time with kw_only fields in
        # dataclasses

        # pylint: disable=unexpected-keyword-arg
        func_def = pt.function.FunctionDefinition(
            parameters=frozenset(input_id_to_name_in_function.values()),
            return_type=ret_type,
            returns=Map(returns),
            tags=self.tags,
        )

        call_parameters = {input_id_to_name_in_function[arg_id]: arg
                           for arg_id, arg in arg_id_to_arg.items()}
        call_site_output = func_def(**call_parameters)

        if isinstance(output, pt.Array):
            return call_site_output
        elif is_array_container_type(output.__class__):
            def _pack_into_container(key, ary):
                key = _get_output_arg_id_str(key)
                return call_site_output[key]

            call_site_output_as_container = rec_keyed_map_array_container(
                _pack_into_container,
                output)
            return call_site_output_as_container
        else:
            raise NotImplementedError(type(output))


# vim: foldmethod=marker
