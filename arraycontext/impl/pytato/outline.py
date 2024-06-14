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
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional, Tuple, Union

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
                if np.isscalar(ary):
                    pass
                else:
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


def _get_input_arg_id_str(arg_id: Tuple[Any, ...]) -> str:
    from arraycontext.impl.pytato.utils import _ary_container_key_stringifier
    return f"_actx_in_{_ary_container_key_stringifier(arg_id)}"


def _get_output_arg_id_str(arg_id: Tuple[Any, ...]) -> str:
    from arraycontext.impl.pytato.utils import _ary_container_key_stringifier
    return f"_actx_out_{_ary_container_key_stringifier(arg_id)}"


def _get_arg_id_to_placeholder(
        arg_id_to_arg: Mapping[Tuple[Any, ...], Any],
    ) -> Map[Tuple[Any, ...], pt.Placeholder]:
    """
    Helper for :meth:`OulinedCall.__call__`. Constructs a :class:`pytato.Placeholder`
    for each argument in *arg_id_to_arg*. See
    :attr:`CompiledFunction.input_id_to_name_in_function` for argument-id's
    representation.
    """
    return Map({
        arg_id: pt.make_placeholder(
            _get_input_arg_id_str(arg_id),
            arg.shape,
            arg.dtype)
        for arg_id, arg in arg_id_to_arg.items()})


def _call_with_placeholders(
        f: Callable[..., Any],
        args: Tuple[Any],
        kwargs: Mapping[str, Any],
        arg_id_to_placeholder: Mapping[Tuple[Any, ...], pt.Placeholder]) -> Any:
    """
    Construct placeholders analogous to *args* and *kwargs* and call *f*.
    """
    def get_placeholder_replacement(arg, key):
        if np.isscalar(arg):
            return arg
        elif isinstance(arg, pt.Array):
            return arg_id_to_placeholder[key]
        elif is_array_container_type(arg.__class__):
            def _rec_to_placeholder(keys, ary):
                return get_placeholder_replacement(ary, key + keys)

            return rec_keyed_map_array_container(_rec_to_placeholder, arg)
        else:
            raise NotImplementedError(type(arg))

    pl_args = [get_placeholder_replacement(arg, (iarg,))
               for iarg, arg in enumerate(args)]
    pl_kwargs = {kw: get_placeholder_replacement(arg, (kw,))
                 for kw, arg in kwargs.items()}

    return f(*pl_args, **pl_kwargs)


def _unpack_output(
        output: ArrayOrContainer) -> Union[pt.Array, Dict[str, pt.Array]]:
    """Unpack any array containers in *output*."""
    if isinstance(output, pt.Array):
        return output
    elif is_array_container_type(output.__class__):
        unpacked_output = {}

        def _unpack_container(key, ary):
            key = _get_output_arg_id_str(key)
            unpacked_output[key] = ary
            return ary

        rec_keyed_map_array_container(_unpack_container, output)

        return unpacked_output
    else:
        raise NotImplementedError(type(output))


def _pack_output(
        output_template: ArrayOrContainer,
        unpacked_output: Union[
            pt.Array,
            Tuple[pt.Array, ...],
            Mapping[str, pt.Array]]
        ) -> ArrayOrContainer:
    """
    Pack *unpacked_output* into array containers according to *output_template*.
    """
    if isinstance(output_template, pt.Array):
        return unpacked_output
    elif is_array_container_type(output_template.__class__):
        def _pack_into_container(key, ary):
            key = _get_output_arg_id_str(key)
            return unpacked_output[key]

        return rec_keyed_map_array_container(_pack_into_container, output_template)
    else:
        raise NotImplementedError(type(output_template))


@dataclass(frozen=True)
class OutlinedCall:
    actx: _BasePytatoArrayContext
    f: Callable[..., Any]
    tags: FrozenSet[Tag]

    def __call__(self, *args: Any, **kwargs: Any) -> ArrayOrContainer:
        arg_id_to_arg = _get_arg_id_to_arg(args, kwargs)

        arg_id_to_placeholder = _get_arg_id_to_placeholder(arg_id_to_arg)

        output = _call_with_placeholders(self.f, args, kwargs, arg_id_to_placeholder)
        unpacked_output = _unpack_output(output)
        if isinstance(unpacked_output, pt.Array):
            unpacked_output = {"_": unpacked_output}
            ret_type = pt.function.ReturnType.ARRAY
        else:
            ret_type = pt.function.ReturnType.DICT_OF_ARRAYS

        call_bindings = {
            placeholder.name: arg_id_to_arg[arg_id]
            for arg_id, placeholder in arg_id_to_placeholder.items()}

        # pylint-disable-reason: pylint has a hard time with kw_only fields in
        # dataclasses

        # pylint: disable=unexpected-keyword-arg
        func_def = pt.function.FunctionDefinition(
            parameters=frozenset(call_bindings.keys()),
            return_type=ret_type,
            returns=Map(unpacked_output),
            tags=self.tags,
        )

        call_site_output = func_def(**call_bindings)

        return _pack_output(output, call_site_output)


# vim: foldmethod=marker
