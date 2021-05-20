"""
.. currentmodule:: arraycontext


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
from typing import Any, Iterable, Tuple, Optional
import numpy as np


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
    the control of the container class.

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
    raise NotImplementedError(type(ary).__name__)


@singledispatch
def deserialize_container(template, iterable: Iterable[Tuple[Any, Any]]):
    """Deserialize an iterable into an array container.

    :param template: an instance of an existing object that
        can be used to aid in the deserialization. For a similar choice
        see :attr:`~numpy.class.__array_finalize__`.
    :param iterable: an iterable that mirrors the output of
        :meth:`serialize_container`.
    """
    raise NotImplementedError(type(template).__name__)


def is_array_container_type(cls: type) -> bool:
    """
    :returns: *True* if the type *cls* has a registered implementation of
        :func:`serialize_container`, or if it is an :class:`ArrayContainer`.
    """
    return (
            cls is ArrayContainer
            or (serialize_container.dispatch(cls)
                is not serialize_container.__wrapped__))


def is_array_container(ary: Any) -> bool:
    """
    :returns: *True* if the instance *ary* has a registered implementation of
        :func:`serialize_container`.
    """
    return (serialize_container.dispatch(ary.__class__)
            is not serialize_container.__wrapped__)


@singledispatch
def get_container_context(ary: ArrayContainer) -> Optional["ArrayContext"]:
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
                f"only object arrays are supported, given dtype '{ary.dtype}'")

    return np.ndenumerate(ary)


@deserialize_container.register(np.ndarray)
def _deserialize_ndarray_container(
        template: Any, iterable: Iterable[Tuple[Any, Any]]):
    # disallow subclasses
    assert type(template) is np.ndarray
    assert template.dtype.char == "O"

    result = type(template)(template.shape, dtype=object)
    for i, subary in iterable:
        result[i] = subary

    return result

# }}}


# {{{ get_container_context_recursively

def get_container_context_recursively(ary) -> Optional["ArrayContext"]:
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


# vim: foldmethod=marker
