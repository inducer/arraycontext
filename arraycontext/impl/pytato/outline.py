from __future__ import annotations


__doc__ = """
.. autoclass:: OutlinedCall
"""
__copyright__ = """
Copyright (C) 2023-5 University of Illinois Board of Trustees
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
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import numpy as np
from immutabledict import immutabledict

import pytato as pt

from arraycontext.container import SerializationKey, is_array_container_type
from arraycontext.container.traversal import rec_keyed_map_array_container
from arraycontext.context import P
from arraycontext.typing import (
    ArrayOrContainerOrScalar,
    is_scalar_like,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pymbolic import Scalar
    from pytools.tag import Tag

    from arraycontext.impl.pytato import _BasePytatoArrayContext
    from arraycontext.typing import (
        Array,
        ArrayOrScalar,
    )


def _get_arg_id_to_arg(
            args: tuple[ArrayOrContainerOrScalar | None, ...],
            kwargs: Mapping[str, ArrayOrContainerOrScalar | None]
        ) -> immutabledict[tuple[SerializationKey, ...], pt.Array]:
    """
    Helper for :meth:`OulinedCall.__call__`. Extracts mappings from argument id
    to argument values. See
    :attr:`CompiledFunction.input_id_to_name_in_function` for argument-id's
    representation.
    """
    arg_id_to_arg: dict[tuple[SerializationKey, ...], object] = {}

    for kw, arg in itertools.chain(enumerate(args),
                                   kwargs.items()):
        if arg is None:
            pass
        elif is_scalar_like(arg):
            # do not make scalars into placeholders since we inline them.
            pass
        elif is_array_container_type(arg.__class__):
            def id_collector(
                        keys: tuple[SerializationKey, ...],
                        ary: ArrayOrScalar
                    ) -> ArrayOrScalar:
                if is_scalar_like(ary):
                    pass
                else:
                    arg_id = (kw, *keys)  # noqa: B023
                    arg_id_to_arg[arg_id] = ary
                return ary

            rec_keyed_map_array_container(id_collector, arg)
        elif isinstance(arg, pt.Array):
            arg_id = (kw,)
            arg_id_to_arg[arg_id] = arg
        else:
            raise ValueError("Argument to a compiled operator should be"
                             " either a scalar, pt.Array or an array container. Got"
                             f" '{arg}'.")

    return immutabledict(arg_id_to_arg)


def _get_input_arg_id_str(
        arg_id: tuple[object, ...], prefix: str | None = None) -> str:
    if prefix is None:
        prefix = ""
    from arraycontext.impl.pytato.utils import _ary_container_key_stringifier
    return f"_actx_{prefix}_in_{_ary_container_key_stringifier(arg_id)}"


def _get_output_arg_id_str(arg_id: tuple[object, ...]) -> str:
    from arraycontext.impl.pytato.utils import _ary_container_key_stringifier
    return f"_actx_out_{_ary_container_key_stringifier(arg_id)}"


def _get_arg_id_to_placeholder(
            arg_id_to_arg: Mapping[tuple[SerializationKey, ...], pt.Array],
            prefix: str | None = None
        ) -> immutabledict[tuple[SerializationKey, ...], pt.Placeholder]:
    """
    Helper for :meth:`OulinedCall.__call__`. Constructs a :class:`pytato.Placeholder`
    for each argument in *arg_id_to_arg*. See
    :attr:`CompiledFunction.input_id_to_name_in_function` for argument-id's
    representation.
    """
    return immutabledict({
        arg_id: pt.make_placeholder(
            _get_input_arg_id_str(arg_id, prefix=prefix),
            arg.shape,
            arg.dtype)
        for arg_id, arg in arg_id_to_arg.items()})


def _call_with_placeholders(
            f: Callable[..., ArrayOrContainerOrScalar],
            args: tuple[ArrayOrContainerOrScalar | None, ...],
            kwargs: Mapping[str, ArrayOrContainerOrScalar | None],
            arg_id_to_placeholder: Mapping[
                tuple[SerializationKey, ...],
                pt.Placeholder]
        ) -> ArrayOrContainerOrScalar:
    """
    Construct placeholders analogous to *args* and *kwargs* and call *f*.
    """
    def get_placeholder_replacement(
                arg: ArrayOrContainerOrScalar | None,
                key: tuple[SerializationKey, ...]
            ) -> ArrayOrContainerOrScalar | None:
        if arg is None:
            return None
        elif np.isscalar(arg):
            return cast("Scalar", arg)
        elif isinstance(arg, pt.Array):
            return arg_id_to_placeholder[key]
        elif is_array_container_type(arg.__class__):
            def _rec_to_placeholder(
                        keys: tuple[SerializationKey, ...],
                        ary: ArrayOrScalar,
                    ) -> ArrayOrScalar:
                return cast("Array", get_placeholder_replacement(ary, key + keys))

            return rec_keyed_map_array_container(_rec_to_placeholder, arg)
        else:
            raise NotImplementedError(type(arg))

    pl_args = [get_placeholder_replacement(arg, (iarg,))
               for iarg, arg in enumerate(args)]
    pl_kwargs = {kw: get_placeholder_replacement(arg, (kw,))
                 for kw, arg in kwargs.items()}

    return f(*pl_args, **pl_kwargs)


def _unpack_output(
        output: ArrayOrContainerOrScalar) -> immutabledict[str, pt.Array]:
    """Unpack any array containers in *output*."""
    if isinstance(output, pt.Array):
        return immutabledict({"_": output})
    elif is_array_container_type(output.__class__):
        unpacked_output = {}

        def _unpack_container(
                    key: tuple[SerializationKey, ...],
                    ary: ArrayOrScalar
                ) -> ArrayOrScalar:
            key_str = _get_output_arg_id_str(key)
            unpacked_output[key_str] = ary
            return ary

        rec_keyed_map_array_container(_unpack_container, output)

        return immutabledict(unpacked_output)
    else:
        raise NotImplementedError(type(output))


def _pack_output(
            output_template: ArrayOrContainerOrScalar,
            unpacked_output: pt.Array | immutabledict[str, pt.Array]
        ) -> ArrayOrContainerOrScalar:
    """
    Pack *unpacked_output* into array containers according to *output_template*.
    """
    if isinstance(output_template, pt.Array):
        assert isinstance(unpacked_output, pt.Array)
        return unpacked_output
    elif is_array_container_type(output_template.__class__):
        assert isinstance(unpacked_output, immutabledict)

        def _pack_into_container(
                    key: tuple[SerializationKey, ...],
                    ary: ArrayOrScalar  # pyright: ignore[reportUnusedParameter]
                ) -> ArrayOrScalar:
            key_str = _get_output_arg_id_str(key)
            return unpacked_output[key_str]

        return rec_keyed_map_array_container(_pack_into_container, output_template)
    else:
        raise NotImplementedError(type(output_template))


OutlinedResultT = TypeVar("OutlinedResultT", bound=ArrayOrContainerOrScalar)


@dataclass(frozen=True)
class OutlinedCall(Generic[P, OutlinedResultT]):
    actx: _BasePytatoArrayContext
    f: Callable[P, OutlinedResultT]
    tags: frozenset[Tag]

    def __call__(self,
                *args: ArrayOrContainerOrScalar | None,
                **kwargs: ArrayOrContainerOrScalar | None,
            ) -> ArrayOrContainerOrScalar:
        arg_id_to_arg = _get_arg_id_to_arg(args, kwargs)

        if __debug__:
            # Function arguments may produce corresponding placeholders that have
            # the same names as placeholders in the parent context. To avoid potential
            # ambiguity, forbid capturing non-argument placeholders in the function
            # body.

            # Add a prefix to the names to distinguish them from any existing
            # placeholders
            arg_id_to_prefixed_placeholder = _get_arg_id_to_placeholder(
                arg_id_to_arg, prefix="outlined_call")

            prefixed_output = _call_with_placeholders(
                self.f, args, kwargs, arg_id_to_prefixed_placeholder)

            unpacked_prefixed_output = pt.transform.deduplicate(
                pt.make_dict_of_named_arrays(_unpack_output(prefixed_output)))

            prefixed_placeholders = frozenset(
                arg_id_to_prefixed_placeholder.values())

            found_placeholders = frozenset({
                arg for arg in pt.transform.InputGatherer()(unpacked_prefixed_output)
                if isinstance(arg, pt.Placeholder)})

            extra_placeholders = found_placeholders - prefixed_placeholders
            assert not extra_placeholders, \
                "Found non-argument placeholder " \
                f"'{next(iter(extra_placeholders)).name}' in outlined function."

        arg_id_to_placeholder = _get_arg_id_to_placeholder(arg_id_to_arg)

        output = _call_with_placeholders(self.f, args, kwargs, arg_id_to_placeholder)
        unpacked_output = pt.transform.deduplicate(
            pt.make_dict_of_named_arrays(_unpack_output(output)))
        if len(unpacked_output) == 1 and "_" in unpacked_output:
            ret_type = pt.function.ReturnType.ARRAY
        else:
            ret_type = pt.function.ReturnType.DICT_OF_ARRAYS

        used_placeholders = frozenset({
            arg for arg in pt.transform.InputGatherer()(unpacked_output)
            if isinstance(arg, pt.Placeholder)})

        call_bindings = {
            placeholder.name: arg_id_to_arg[arg_id]
            for arg_id, placeholder in arg_id_to_placeholder.items()
            if placeholder in used_placeholders}

        func_def = pt.function.FunctionDefinition(
            parameters=frozenset(call_bindings.keys()),
            return_type=ret_type,
            returns=immutabledict(unpacked_output._data),
            tags=self.tags,
        )

        call_site_output = func_def(**call_bindings)

        assert isinstance(call_site_output, pt.Array | immutabledict)
        return _pack_output(output, call_site_output)

# vim: foldmethod=marker
