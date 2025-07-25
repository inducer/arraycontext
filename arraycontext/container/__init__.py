"""
.. currentmodule:: arraycontext

.. class:: ArrayContainer
    A protocol for generic containers of the array type supported by the
    :class:`ArrayContext`.

    The functionality required for the container to operated is supplied via
    :func:`functools.singledispatch`. Implementations of the following functions need
    to be registered for a type serving as an :class:`ArrayContainer`:

    * :func:`serialize_container` for serialization, which gives the components
      of the array.
    * :func:`deserialize_container` for deserialization, which constructs a
      container from a set of components.
    * :func:`get_container_context_opt` retrieves the :class:`ArrayContext` from
      a container, if it has one.

    This allows enumeration of the component arrays in a container and the
    construction of modified containers from an iterable of those component arrays.

    Packages may register their own types as array containers. They must not
    register other types (e.g. :class:`list`) as array containers.
    The type :class:`numpy.ndarray` is considered an array container, but
    only arrays with dtype *object* may be used as such. (This is so
    because object arrays cannot be distinguished from non-object arrays
    via their type.)

    The container and its serialization interface has goals and uses
    approaches similar to JAX's
    `PyTrees <https://jax.readthedocs.io/en/latest/pytrees.html>`__,
    however its implementation differs a bit.

    .. note::

        This class is used in type annotation and as a marker of array container
        attributes for :func:`~arraycontext.dataclass_array_container`.
        As a protocol, it is not intended as a superclass.

    .. note::

        For the benefit of type checkers, array containers are recognized by
        having the declaration::

            __array_ufunc__: ClassVar[None] = None

        in their body. In addition to its use as a recognition feature, this also
        prevents unintended arithmetic in conjunction with :mod:`numpy` arrays.
        This should be considered experimental for now, and it may well change.

.. autoclass:: ArithArrayContainer
.. autoclass:: ArrayContainerT
.. autoclass:: ArithArrayContainerT

.. autoexception:: NotAnArrayContainerError

Serialization/deserialization
-----------------------------

.. autoclass:: SerializationKey
.. autoclass:: SerializedContainer
.. autofunction:: is_array_container_type
.. autofunction:: serialize_container
.. autofunction:: deserialize_container

Context retrieval
-----------------
.. autofunction:: get_container_context_opt
.. autofunction:: get_container_context_recursively
.. autofunction:: get_container_context_recursively_opt

:class:`~pymbolic.geometric_algebra.MultiVector` support
---------------------------------------------------------

.. autofunction:: register_multivector_as_array_container

.. currentmodule:: arraycontext.container

Canonical locations for type annotations
----------------------------------------

.. class:: ArrayContainerT

    :canonical: arraycontext.ArrayContainerT

.. class:: SerializationKey

    :canonical: arraycontext.SerializationKey

.. class:: SerializedContainer

    :canonical: arraycontext.SerializedContainer

References
----------

.. class:: GenericAlias

    See :class:`types.GenericAlias`.

.. class:: UnionType

    See :class:`types.UnionType`.
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

from collections.abc import Hashable, Sequence
from functools import singledispatch
from typing import (
    TYPE_CHECKING,
    TypeAlias,
)

# For use in singledispatch type annotations, because sphinx can't figure out
# what 'np' is.
import numpy
import numpy as np
from typing_extensions import TypeIs

from pytools.obj_array import ObjectArray, ObjectArrayND as ObjectArrayND

from arraycontext.typing import (
    ArithArrayContainer as ArithArrayContainer,
    ArrayContainer,
    ArrayContainerT,
    ArrayOrArithContainer,
    ArrayOrArithContainerOrScalar as ArrayOrArithContainerOrScalar,
    ArrayOrContainerOrScalar,
    _UserDefinedArithArrayContainer,
    _UserDefinedArrayContainer,
    all_type_leaves_satisfy_predicate,
)


if TYPE_CHECKING:
    from types import GenericAlias, UnionType

    from pymbolic.geometric_algebra import CoeffT, MultiVector

    from arraycontext.context import ArrayContext
    from arraycontext.typing import ArrayOrScalar as ArrayOrScalar


# {{{ ArrayContainer traversals

class NotAnArrayContainerError(TypeError):
    """:class:`TypeError` subclass raised when an array container is expected."""


SerializationKey: TypeAlias = Hashable
SerializedContainer: TypeAlias = Sequence[
    tuple[SerializationKey, ArrayOrContainerOrScalar]]


@singledispatch
def serialize_container(
        ary: ArrayContainer) -> SerializedContainer:
    r"""Serialize the array container into a sequence over its components.

    The order of the components and their identifiers are entirely under
    the control of the container class. However, the order is required to be
    deterministic, i.e. two calls to :func:`serialize_container` on
    array containers of the same types with the same number of
    sub-arrays must result in a sequence with the keys in the same
    order.

    If *ary* is mutable, the serialization function is not required to ensure
    that the serialization result reflects the array state at the time of the
    call to :func:`serialize_container`.

    :returns: a :class:`Sequence` of 2-tuples where the first
        entry is an identifier for the component and the second entry
        is an array-like component of the :class:`ArrayContainer`.
        Components can themselves be :class:`ArrayContainer`\ s, allowing
        for arbitrarily nested structures. The identifiers need to be hashable
        but are otherwise treated as opaque.
    """
    raise NotAnArrayContainerError(
            f"'{type(ary).__name__}' cannot be serialized as a container")


@singledispatch
def deserialize_container(
        template: ArrayContainerT,
        serialized: SerializedContainer) -> ArrayContainerT:
    """Deserialize a sequence into an array container following a *template*.

    :param template: an instance of an existing object that
        can be used to aid in the deserialization. For a similar choice
        see :attr:`~numpy.class.__array_finalize__`.
    :param serialized: a sequence that mirrors the output of
        :meth:`serialize_container`.
    """
    raise NotAnArrayContainerError(
            f"'{type(template).__name__}' cannot be deserialized as a container")


def is_array_container_type(cls: type | GenericAlias | UnionType) -> bool:
    """
    :returns: *True* if the type *cls* has a registered implementation of
        :func:`serialize_container`, or if it is an :class:`ArrayContainer`.

    .. warning::

        Not all instances of a type that this function labels an array container
        must automatically be array containers. For example, while this
        function will say that :class:`numpy.ndarray` is an array container
        type, only object arrays *actually are* array containers.
    """
    def _is_array_container_type(tp: type) -> bool:
        return (
                tp is ObjectArray
                or tp is _UserDefinedArrayContainer
                or tp is _UserDefinedArithArrayContainer
                or (serialize_container.dispatch(tp)
                    is not serialize_container.__wrapped__))  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    return all_type_leaves_satisfy_predicate(_is_array_container_type, cls)


def is_array_container(ary: object) -> TypeIs[ArrayContainer]:
    """
    :returns: *True* if the instance *ary* has a registered implementation of
        :func:`serialize_container`.
    """

    from warnings import warn
    warn("is_array_container is deprecated and will be removed in 2022. "
            "If you must know precisely whether something is an array container, "
            "try serializing it and catch NotAnArrayContainerError. For a "
            "cheaper option, see is_array_container_type.",
            DeprecationWarning, stacklevel=2)
    return (serialize_container.dispatch(ary.__class__)
            is not serialize_container.__wrapped__       # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            # numpy values with scalar elements aren't array containers
            and not (isinstance(ary, np.ndarray)
                     and ary.dtype.kind != "O")
            )


@singledispatch
def get_container_context_opt(ary: ArrayContainer) -> ArrayContext | None:
    """Retrieves the :class:`ArrayContext` from the container, if any.

    This function is not recursive, so it will only search at the root level
    of the container. For the recursive version, see
    :func:`get_container_context_recursively`.
    """
    return getattr(ary, "array_context", None)

# }}}


# {{{ object arrays as array containers

# Sadly, ObjectArray is not usable here.
@serialize_container.register(np.ndarray)
def _serialize_ndarray_container(ary: numpy.ndarray) -> SerializedContainer:
    if ary.dtype.char != "O":
        raise NotAnArrayContainerError(
                f"cannot serialize '{type(ary).__name__}' with dtype '{ary.dtype}'")

    # special-cased for speed
    if ary.ndim == 1:
        return [(i, ary[i]) for i in range(ary.shape[0])]
    elif ary.ndim == 2:
        return [((i, j), ary[i, j])
                for i in range(ary.shape[0])
                for j in range(ary.shape[1])
                ]
    else:
        return list(np.ndenumerate(ary))


@deserialize_container.register(np.ndarray)
# https://github.com/python/mypy/issues/13040
def _deserialize_ndarray_container(  # type: ignore[misc]
        template: numpy.ndarray,
        serialized: SerializedContainer) -> numpy.ndarray:
    # disallow subclasses
    assert type(template) is np.ndarray
    assert template.dtype.char == "O"

    result = type(template)(template.shape, dtype=object)
    for i, subary in serialized:
        # FIXME: numpy annotations don't seem to handle object arrays very well
        result[i] = subary  # type: ignore[call-overload]

    return result

# }}}


# {{{ get_container_context_recursively

def get_container_context_recursively_opt(
        ary: ArrayOrContainerOrScalar) -> ArrayContext | None:
    """Walks the :class:`ArrayContainer` hierarchy to find an
    :class:`ArrayContext` associated with it.

    If different components that have different array contexts are found at
    any level, an assertion error is raised.

    Returns *None* if no array context was found.
    """
    # try getting the array context directly
    actx = get_container_context_opt(ary)
    if actx is not None:
        return actx

    try:
        iterable = serialize_container(ary)
    except NotAnArrayContainerError:
        return actx
    else:
        for _, subary in iterable:
            context = get_container_context_recursively_opt(subary)
            if context is None:
                continue

            if not __debug__:
                return context
            elif actx is None:
                actx = context
            else:
                assert actx is context

        return actx


def get_container_context_recursively(ary: ArrayContainer) -> ArrayContext:
    """Walks the :class:`ArrayContainer` hierarchy to find an
    :class:`ArrayContext` associated with it.

    If different components that have different array contexts are found at
    any level, an assertion error is raised.

    Raises an error if no array container is found.
    """
    actx = get_container_context_recursively_opt(ary)
    if actx is None:
        raise ValueError("no array context was found")

    return actx

# }}}


# {{{ MultiVector support, see pymbolic.geometric_algebra

# FYI: This doesn't, and never should, make arraycontext directly depend on pymbolic.
# (Though clearly there exists a dependency via loopy.)

def _serialize_multivec_as_container(
            mv: MultiVector[ArrayOrArithContainer]
        ) -> SerializedContainer:
    return list(mv.data.items())


def _deserialize_multivec_as_container(
        template: MultiVector[CoeffT],
        serialized: SerializedContainer) -> MultiVector[CoeffT]:
    from pymbolic.geometric_algebra import MultiVector
    return MultiVector(dict(serialized), space=template.space)


def _get_container_context_opt_from_multivec(mv: MultiVector[CoeffT]) -> None:
    return None


def register_multivector_as_array_container() -> None:
    """Registers :class:`~pymbolic.geometric_algebra.MultiVector` as an
    :class:`ArrayContainer`.  This function may be called multiple times. The
    second and subsequent calls have no effect.
    """
    from pymbolic.geometric_algebra import MultiVector
    if MultiVector not in serialize_container.registry:
        serialize_container.register(MultiVector)(_serialize_multivec_as_container)
        deserialize_container.register(MultiVector)(
                _deserialize_multivec_as_container)
        get_container_context_opt.register(MultiVector)(
                _get_container_context_opt_from_multivec)
        assert MultiVector in serialize_container.registry

# }}}


# vim: foldmethod=marker
