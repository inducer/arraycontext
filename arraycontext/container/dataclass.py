"""
.. currentmodule:: arraycontext
.. autofunction:: dataclass_array_container

References
----------

.. currentmodule:: arraycontext.container.dataclass
.. class:: T

    A type variable. Represents the dataclass being turned into an array container.
"""
from __future__ import annotations


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

from dataclasses import fields, is_dataclass
from typing import (
    TYPE_CHECKING,
    NamedTuple,
    TypeVar,
)
from warnings import warn

import numpy as np

from arraycontext.container import is_array_container_type
from arraycontext.typing import (
    all_type_leaves_satisfy_predicate,
    is_scalar_type,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import GenericAlias, UnionType


T = TypeVar("T")


# {{{ dataclass containers

class _Field(NamedTuple):
    """Small lookalike for :class:`dataclasses.Field`."""

    init: bool
    name: str
    type: type


def _is_array_or_container_type(
            tp: type | GenericAlias | UnionType | TypeVar, /, *,
            allow_scalar: bool = True,
            require_homogeneity: bool = True,
        ) -> bool:
    def _is_array_or_container_or_scalar(tp: type) -> bool:
        if tp is np.ndarray:
            warn("Encountered 'numpy.ndarray' in a dataclass_array_container. "
                 "This is deprecated and will stop working in 2026. "
                 "If you meant an object array, use pytools.obj_array.ObjectArray. "
                 "For other uses, file an issue to discuss.",
                 DeprecationWarning, stacklevel=1)
            return True

        from arraycontext import Array

        return (
                is_array_container_type(tp)
                or tp is Array
                or (allow_scalar and is_scalar_type(tp)))

    return all_type_leaves_satisfy_predicate(
                    _is_array_or_container_or_scalar, tp,
                    require_homogeneity=require_homogeneity)


def dataclass_array_container(cls: type[T]) -> type[T]:
    """A class decorator that makes the class to which it is applied an
    :class:`ArrayContainer` by registering appropriate implementations of
    :func:`serialize_container` and :func:`deserialize_container`.
    *cls* must be a :func:`~dataclasses.dataclass`.

    Attributes that are not array containers are allowed. In order to decide
    whether an attribute is an array container, the declared attribute type
    is checked by the criteria from :func:`is_array_container_type`. This
    includes some support for type annotations:

    * a :class:`typing.Union` of array containers is considered an array container.
    * other type annotations, e.g. :class:`typing.Optional`, are not considered
      array containers, even if they wrap one.

    .. note::

        When type annotations are strings (e.g. because of
        ``from __future__ import annotations``),
        this function relies on :func:`inspect.get_annotations`
        (with ``eval_str=True``) to obtain type annotations. This
        means that *cls* must live in a module that is importable.
    """

    assert is_dataclass(cls)

    def is_array_field(f: _Field) -> bool:
        field_type = f.type
        assert not isinstance(field_type, str)

        if not f.init:
            raise ValueError(
                    f"Field with 'init=False' not allowed: '{f.name}'")

        try:
            return _is_array_or_container_type(field_type)
        except TypeError as e:
            raise TypeError(f"Field '{f.name}': {e}") from None

    from pytools import partition

    array_fields = _get_annotated_fields(cls)
    array_fields, non_array_fields = partition(is_array_field, array_fields)

    if not array_fields:
        raise ValueError(f"'{cls}' must have fields with array container type "
                "in order to use the 'dataclass_array_container' decorator")

    return _inject_dataclass_serialization(cls, array_fields, non_array_fields)


def _get_annotated_fields(cls: type) -> Sequence[_Field]:
    """Get a list of fields in the class *cls* with evaluated types.

    If any of the fields in *cls* have type annotations that are strings, e.g.
    from using ``from __future__ import annotations``, this function evaluates
    them using :func:`inspect.get_annotations`. Note that this requires the class
    to live in a module that is importable.

    :return: a list of fields.
    """

    from inspect import get_annotations

    result = []
    field_name_to_type: Mapping[str, type] | None = None
    for field in fields(cls):
        field_type_or_str = field.type
        if isinstance(field_type_or_str, str):
            if field_name_to_type is None:
                field_name_to_type = {}
                for subcls in cls.__mro__[::-1]:
                    field_name_to_type.update(get_annotations(subcls, eval_str=True))

            field_type = field_name_to_type[field.name]
        else:
            field_type = field_type_or_str

        result.append(_Field(init=field.init, name=field.name, type=field_type))

    return result


def _inject_dataclass_serialization(
        cls: type,
        array_fields: Sequence[_Field],
        non_array_fields: Sequence[_Field]) -> type:
    """Implements :func:`~arraycontext.serialize_container` and
    :func:`~arraycontext.deserialize_container` for the given dataclass *cls*.

    This function modifies *cls* in place, so the returned value is the same
    object with additional functionality.

    :arg array_fields: fields of the given dataclass *cls* which are considered
        array containers and should be serialized.
    :arg non_array_fields: remaining fields of the dataclass *cls* which are
        copied over from the template array in deserialization.
    """

    assert is_dataclass(cls)

    serialize_expr = ", ".join(
            f"({f.name!r}, ary.{f.name})" for f in array_fields)
    template_kwargs = ", ".join(
            f"{f.name}=template.{f.name}" for f in non_array_fields)

    lower_cls_name = cls.__name__.lower()

    serialize_init_code = ", ".join(f"{f.name!r}: f'{{instance_name}}.{f.name}'"
            for f in array_fields)
    deserialize_init_code = ", ".join([
            f"{f.name}={{args[{f.name!r}]}}" for f in array_fields
            ] + [
            f"{f.name}={{template_instance_name}}.{f.name}"
            for f in non_array_fields
            ])

    from pytools.codegen import remove_common_indentation
    serialize_code = remove_common_indentation(f"""
        from typing import Any, Iterable, Tuple
        from arraycontext import serialize_container, deserialize_container

        @serialize_container.register(cls)
        def _serialize_{lower_cls_name}(ary: cls) -> Iterable[Tuple[Any, Any]]:
            return ({serialize_expr},)

        @deserialize_container.register(cls)
        def _deserialize_{lower_cls_name}(
                template: cls, iterable: Iterable[Tuple[Any, Any]]) -> cls:
            return cls(**dict(iterable), {template_kwargs})

        # support for with_container_arithmetic

        def _serialize_init_arrays_code_{lower_cls_name}(cls, instance_name):
            return {{
                {serialize_init_code}
                }}
        cls._serialize_init_arrays_code = classmethod(
            _serialize_init_arrays_code_{lower_cls_name})

        def _deserialize_init_arrays_code_{lower_cls_name}(
                cls, template_instance_name, args):
            return f"{deserialize_init_code}"

        cls._deserialize_init_arrays_code = classmethod(
            _deserialize_init_arrays_code_{lower_cls_name})
        """)

    exec_dict = {"cls": cls, "_MODULE_SOURCE_CODE": serialize_code}
    exec(compile(serialize_code, f"<container serialization for {cls.__name__}>",
        "exec"), exec_dict)

    return cls

# }}}

# vim: foldmethod=marker
