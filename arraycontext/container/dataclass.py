# mypy: disallow-untyped-defs

"""
.. currentmodule:: arraycontext
.. autofunction:: dataclass_array_container
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

from dataclasses import Field, fields, is_dataclass
from typing import Tuple, Union, get_args, get_origin

from arraycontext.container import is_array_container_type


# {{{ dataclass containers

def is_array_type(tp: type) -> bool:
    from arraycontext import Array
    return tp is Array or is_array_container_type(tp)


def dataclass_array_container(cls: type) -> type:
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
    """

    assert is_dataclass(cls)

    def is_array_field(f: Field) -> bool:
        # NOTE: unions of array containers are treated separately to handle
        # unions of only array containers, e.g. `Union[np.ndarray, Array]`, as
        # they can work seamlessly with arithmetic and traversal.
        #
        # `Optional[ArrayContainer]` is not allowed, since `None` is not
        # handled by `with_container_arithmetic`, which is the common case
        # for current container usage. Other type annotations, e.g.
        # `Tuple[Container, Container]`, are also not allowed, as they do not
        # work with `with_container_arithmetic`.
        #
        # This is not set in stone, but mostly driven by current usage!

        origin = get_origin(f.type)
        if origin is Union:
            if all(is_array_type(arg) for arg in get_args(f.type)):
                return True
            else:
                raise TypeError(
                        f"Field '{f.name}' union contains non-array container "
                        "arguments. All arguments must be array containers.")

        if isinstance(f.type, str):
            raise TypeError(
                f"String annotation on field '{f.name}' not supported. "
                "(this may be due to 'from __future__ import annotations')")

        if __debug__:
            if not f.init:
                raise ValueError(
                        f"Field with 'init=False' not allowed: '{f.name}'")

            # NOTE:
            # * `_BaseGenericAlias` catches `List`, `Tuple`, etc.
            # * `_SpecialForm` catches `Any`, `Literal`, etc.
            from typing import (  # type: ignore[attr-defined]
                _BaseGenericAlias,
                _SpecialForm,
            )
            if isinstance(f.type, (_BaseGenericAlias, _SpecialForm)):
                # NOTE: anything except a Union is not allowed
                raise TypeError(
                        f"Typing annotation not supported on field '{f.name}': "
                        f"'{f.type!r}'")

            if not isinstance(f.type, type):
                raise TypeError(
                        f"Field '{f.name}' not an instance of 'type': "
                        f"'{f.type!r}'")

        return is_array_type(f.type)

    from pytools import partition
    array_fields, non_array_fields = partition(is_array_field, fields(cls))

    if not array_fields:
        raise ValueError(f"'{cls}' must have fields with array container type "
                "in order to use the 'dataclass_array_container' decorator")

    return inject_dataclass_serialization(cls, array_fields, non_array_fields)


def inject_dataclass_serialization(
        cls: type,
        array_fields: Tuple[Field, ...],
        non_array_fields: Tuple[Field, ...]) -> type:
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
