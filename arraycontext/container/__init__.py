# mypy: disallow-untyped-defs

"""
.. currentmodule:: arraycontext

.. class:: ArrayT
    :canonical: arraycontext.container.ArrayT

    :class:`~typing.TypeVar` for arrays.

.. class:: ContainerT
    :canonical: arraycontext.container.ContainerT

    :class:`~typing.TypeVar` for array container-like objects.

.. class:: ArrayOrContainerT
    :canonical: arraycontext.container.ArrayOrContainerT

    :class:`~typing.TypeVar` for arrays or array container-like objects.

.. autoclass:: ArrayContainer

Serialization/deserialization
-----------------------------
.. autofunction:: is_array_container
.. autofunction:: serialize_container
.. autofunction:: deserialize_container

Context retrieval
-----------------
.. autofunction:: get_container_context
.. autofunction:: get_container_context_recursively

:class:`~pymbolic.geometric_algebra.MultiVector` support
---------------------------------------------------------

.. autofunction:: register_multivector_as_array_container
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

from functools import singledispatch
from arraycontext.context import ArrayContext
from typing import Any, Iterable, Tuple, TypeVar, Optional, Union, TYPE_CHECKING
import numpy as np

ArrayT = TypeVar("ArrayT")
ContainerT = TypeVar("ContainerT")
ArrayOrContainerT = Union[ArrayT, ContainerT]

if TYPE_CHECKING:
    from pymbolic.geometric_algebra import MultiVector


# {{{ ArrayContainer

class ArrayContainer:
    r"""
    A generic container for the array type supported by the
    :class:`ArrayContext`.

    The functionality required for the container to operated is supplied via
    :func:`functools.singledispatch`. Implementations of the following functions need
    to be registered for a type serving as an :class:`ArrayContainer`:

    * :func:`serialize_container` for serialization, which gives the components
      of the array.
    * :func:`deserialize_container` for deserialization, which constructs a
      container from a set of components.
    * :func:`get_container_context` retrieves the :class:`ArrayContext` from
      a container, if it has one.

    This allows enumeration of the component arrays in a container and the
    construction of modified containers from an iterable of those component arrays.
    :func:`is_array_container` will return *True* for types that have
    a container serialization function registered.

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

        This class is used in type annotation. Inheriting from it confers no
        special meaning or behavior.
    """


@singledispatch
def serialize_container(ary: ArrayContainer) -> Iterable[Tuple[Any, Any]]:
    r"""Serialize the array container into an iterable over its components.

    The order of the components and their identifiers are entirely under
    the control of the container class. However, the order is required to be
    deterministic, i.e. two calls to :func:`serialize_container` on the same
    array container should return an iterable with the components in the same
    order.

    If *ary* is mutable, the serialization function is not required to ensure
    that the serialization result reflects the array state at the time of the
    call to :func:`serialize_container`.

    :returns: an :class:`Iterable` of 2-tuples where the first
        entry is an identifier for the component and the second entry
        is an array-like component of the :class:`ArrayContainer`.
        Components can themselves be :class:`ArrayContainer`\ s, allowing
        for arbitrarily nested structures. The identifiers need to be hashable
        but are otherwise treated as opaque.
    """
    raise TypeError(f"'{type(ary).__name__}' cannot be serialized as a container")


@singledispatch
def deserialize_container(template: Any, iterable: Iterable[Tuple[Any, Any]]) -> Any:
    """Deserialize an iterable into an array container.

    :param template: an instance of an existing object that
        can be used to aid in the deserialization. For a similar choice
        see :attr:`~numpy.class.__array_finalize__`.
    :param iterable: an iterable that mirrors the output of
        :meth:`serialize_container`.
    """
    raise TypeError(
            f"'{type(template).__name__}' cannot be deserialized as a container")


def is_array_container_type(cls: type) -> bool:
    """
    :returns: *True* if the type *cls* has a registered implementation of
        :func:`serialize_container`, or if it is an :class:`ArrayContainer`.
    """
    return (
            cls is ArrayContainer
            or (serialize_container.dispatch(cls)
                is not serialize_container.__wrapped__))  # type:ignore[attr-defined]


def is_array_container(ary: Any) -> bool:
    """
    :returns: *True* if the instance *ary* has a registered implementation of
        :func:`serialize_container`.
    """
    return (serialize_container.dispatch(ary.__class__)
            is not serialize_container.__wrapped__)       # type:ignore[attr-defined]


@singledispatch
def get_container_context(ary: ArrayContainer) -> Optional[ArrayContext]:
    """Retrieves the :class:`ArrayContext` from the container, if any.

    This function is not recursive, so it will only search at the root level
    of the container. For the recursive version, see
    :func:`get_container_context_recursively`.
    """
    return getattr(ary, "array_context", None)

# }}}


# {{{ object arrays as array containers

@serialize_container.register(np.ndarray)
def _serialize_ndarray_container(ary: np.ndarray) -> Iterable[Tuple[Any, Any]]:
    if ary.dtype.char != "O":
        raise ValueError(
                f"cannot seriealize '{type(ary).__name__}' with dtype '{ary.dtype}'")

    # special-cased for speed
    if ary.ndim == 1:
        return [(i, ary[i]) for i in range(ary.shape[0])]
    elif ary.ndim == 2:
        return [((i, j), ary[i, j])
                for i in range(ary.shape[0])
                for j in range(ary.shape[1])
                ]
    else:
        return np.ndenumerate(ary)


@deserialize_container.register(np.ndarray)
def _deserialize_ndarray_container(
        template: np.ndarray,
        iterable: Iterable[Tuple[Any, Any]]) -> np.ndarray:
    # disallow subclasses
    assert type(template) is np.ndarray
    assert template.dtype.char == "O"

    result = type(template)(template.shape, dtype=object)
    for i, subary in iterable:
        result[i] = subary

    return result

# }}}


# {{{ get_container_context_recursively

def get_container_context_recursively(ary: Any) -> Optional[ArrayContext]:
    """Walks the :class:`ArrayContainer` hierarchy to find an
    :class:`ArrayContext` associated with it.

    If different components that have different array contexts are found at
    any level, an assertion error is raised.
    """
    actx = None
    if not is_array_container(ary):
        return actx

    # try getting the array context directly
    actx = get_container_context(ary)
    if actx is not None:
        return actx

    for _, subary in serialize_container(ary):
        context = get_container_context_recursively(subary)
        if context is None:
            continue

        if not __debug__:
            return context
        elif actx is None:
            actx = context
        else:
            assert actx is context

    return actx

# }}}


# {{{ MultiVector support, see pymbolic.geometric_algebra

# FYI: This doesn't, and never should, make arraycontext directly depend on pymbolic.
# (Though clearly there exists a dependency via loopy.)

def _serialize_multivec_as_container(mv: "MultiVector") -> Iterable[Tuple[Any, Any]]:
    return list(mv.data.items())


def _deserialize_multivec_as_container(template: "MultiVector",
        iterable: Iterable[Tuple[Any, Any]]) -> "MultiVector":
    from pymbolic.geometric_algebra import MultiVector
    return MultiVector(dict(iterable), space=template.space)


def _get_container_context_from_multivec(mv: "MultiVector") -> None:
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
        get_container_context.register(MultiVector)(
                _get_container_context_from_multivec)
        assert MultiVector in serialize_container.registry

# }}}


# vim: foldmethod=marker
