"""
.. currentmodule:: arraycontext

.. autofunction:: map_array_container
.. autofunction:: multimap_array_container
.. autofunction:: rec_map_array_container
.. autofunction:: rec_multimap_array_container

Traversing decorators
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: mapped_over_array_containers
.. autofunction:: multimapped_over_array_containers

Freezing and thawing
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: freeze
.. autofunction:: thaw

Numpy conversion
~~~~~~~~~~~~~~~~
.. autofunction:: from_numpy
.. autofunction:: to_numpy
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

from typing import Any, Callable
from functools import update_wrapper, partial, singledispatch

import numpy as np

from arraycontext.container import (is_array_container,
        serialize_container, deserialize_container)


# {{{ array container traversal

def _map_array_container_impl(f, ary, *, leaf_cls=None, recursive=False):
    """Helper for :func:`rec_map_array_container`.

    :param leaf_cls: class on which we call *f* directly. This is mostly
        useful in the recursive setting, where it can stop the recursion on
        specific container classes. By default, the recursion is stopped when
        a non-:class:`ArrayContainer` class is encountered.
    """
    def rec(_ary):
        if type(_ary) is leaf_cls:  # type(ary) is never None
            return f(_ary)
        elif is_array_container(_ary):
            return deserialize_container(_ary, [
                    (key, frec(subary)) for key, subary in serialize_container(_ary)
                    ])
        else:
            return f(_ary)

    frec = rec if recursive else f
    return rec(ary)


def _multimap_array_container_impl(f, *args, leaf_cls=None, recursive=False):
    """Helper for :func:`rec_multimap_array_container`.

    :param leaf_cls: class on which we call *f* directly. This is mostly
        useful in the recursive setting, where it can stop the recursion on
        specific container classes. By default, the recursion is stopped when
        a non-:class:`ArrayContainer` class is encountered.
    """
    def rec(*_args):
        template_ary = _args[container_indices[0]]
        assert all(
                type(_args[i]) is type(template_ary) for i in container_indices[1:]
                ), "expected type '{type(template_ary).__name__}'"

        if (type(template_ary) is leaf_cls
                or not is_array_container(template_ary)):
            return f(*_args)

        result = []
        new_args = list(_args)

        for subarys in zip(*[
                serialize_container(_args[i]) for i in container_indices
                ]):
            key = None
            for i, (subkey, subary) in zip(container_indices, subarys):
                if key is None:
                    key = subkey
                else:
                    assert key == subkey

                new_args[i] = subary

            result.append((key, frec(*new_args)))

        return deserialize_container(template_ary, result)

    container_indices = [
            i for i, arg in enumerate(args)
            if is_array_container(arg) and type(arg) is not leaf_cls]

    if not container_indices:
        return f(*args)

    if len(container_indices) == 1:
        # NOTE: if we just have one ArrayContainer in args, passing it through
        # _map_array_container_impl should be faster
        def wrapper(ary):
            new_args = list(args)
            new_args[container_indices[0]] = ary
            return f(*new_args)

        update_wrapper(wrapper, f)
        return _map_array_container_impl(
                wrapper, args[container_indices[0]],
                leaf_cls=leaf_cls, recursive=recursive)

    frec = rec if recursive else f
    return rec(*args)


def map_array_container(f: Callable[[Any], Any], ary):
    r"""Applies *f* to all components of an :class:`ArrayContainer`.

    Works similarly to :func:`~pytools.obj_array.obj_array_vectorize`, but
    on arbitrary containers.

    For a recursive version, see :func:`rec_map_array_container`.

    :param ary: a (potentially nested) structure of :class:`ArrayContainer`\ s,
        or an instance of a base array type.
    """
    if is_array_container(ary):
        return deserialize_container(ary, [
                (key, f(subary)) for key, subary in serialize_container(ary)
                ])
    else:
        return f(ary)


def multimap_array_container(f: Callable[[Any], Any], *args):
    r"""Applies *f* to the components of multiple :class:`ArrayContainer`\ s.

    Works similarly to :func:`~pytools.obj_array.obj_array_vectorize_n_args`,
    but on arbitrary containers. The containers must all have the same type,
    which will also be the return type.

    For a recursive version, see :func:`rec_multimap_array_container`.

    :param args: all :class:`ArrayContainer` arguments must be of the same
        type and with the same structure (same number of components, etc.).
    """
    return _multimap_array_container_impl(f, *args, recursive=False)


def rec_map_array_container(f: Callable[[Any], Any], ary):
    r"""Applies *f* recursively to an :class:`ArrayContainer`.

    For a non-recursive version see :func:`map_array_container`.

    :param ary: a (potentially nested) structure of :class:`ArrayContainer`\ s,
        or an instance of a base array type.
    """
    return _map_array_container_impl(f, ary, recursive=True)


def mapped_over_array_containers(f: Callable[[Any], Any]):
    """Decorator around :func:`rec_map_array_container`."""
    wrapper = partial(rec_map_array_container, f)
    update_wrapper(wrapper, f)
    return wrapper


def rec_multimap_array_container(f: Callable[[Any], Any], *args):
    r"""Applies *f* recursively to multiple :class:`ArrayContainer`\ s.

    For a non-recursive version see :func:`multimap_array_container`.

    :param args: all :class:`ArrayContainer` arguments must be of the same
        type and with the same structure (same number of components, etc.).
    """
    return _multimap_array_container_impl(f, *args, recursive=True)


def multimapped_over_array_containers(f: Callable[[Any], Any]):
    """Decorator around :func:`rec_multimap_array_container`."""
    # can't use functools.partial, because its result is insufficiently
    # function-y to be used as a method definition.
    def wrapper(*args):
        return rec_multimap_array_container(f, *args)

    update_wrapper(wrapper, f)
    return wrapper

# }}}


# {{{ freeze/thaw

@singledispatch
def freeze(ary, actx=None):
    r"""Freezes recursively by going through all components of the
    :class:`ArrayContainer` *ary*.

    :param ary: a :meth:`~ArrayContext.thaw`\ ed :class:`ArrayContainer`.

    Array container types may use :func:`functools.singledispatch` ``.register`` to
    register additional implementations.

    See :meth:`ArrayContext.thaw`.
    """
    if is_array_container(ary):
        return map_array_container(partial(freeze, actx=actx), ary)
    else:
        if actx is None:
            raise TypeError(
                    f"cannot freeze arrays of type {type(ary).__name__} "
                    "when actx is not supplied. Try calling actx.freeze "
                    "directly or supplying an array context")
        else:
            return actx.freeze(ary)


@singledispatch
def thaw(ary, actx):
    r"""Thaws recursively by going through all components of the
    :class:`ArrayContainer` *ary*.

    :param ary: a :meth:`~ArrayContext.freeze`\ ed :class:`ArrayContainer`.

    Array container types may use :func:`functools.singledispatch` ``.register``
    to register additional implementations.

    See :meth:`ArrayContext.thaw`.

    Serves as the registration point (using :func:`~functools.singledispatch`
    ``.register`` to register additional implementations for :func:`thaw`.

    .. note::

        This function has the reverse argument order from the original function
        in :mod:`meshmode`. This was necessary because
        :func:`~functools.singledispatch` only dispatches on the first argument.
    """
    if is_array_container(ary):
        return deserialize_container(ary, [
            (key, thaw(subary, actx))
            for key, subary in serialize_container(ary)
            ])
    else:
        return actx.thaw(ary)

# }}}


# {{{ numpy conversion

def from_numpy(ary, actx):
    """Convert all :mod:`numpy` arrays in the :class:`~arraycontext.ArrayContainer`
    to the base array type of :class:`~arraycontext.ArrayContext`.

    The conversion is done using :meth:`arraycontext.ArrayContext.from_numpy`.
    """
    def _from_numpy(subary):
        if isinstance(subary, np.ndarray) and subary.dtype != "O":
            return actx.from_numpy(subary)
        elif is_array_container(subary):
            return map_array_container(_from_numpy, subary)
        else:
            raise TypeError(f"unrecognized array type: '{type(subary).__name__}'")

    return _from_numpy(ary)


def to_numpy(ary, actx):
    """Convert all arrays in the :class:`~arraycontext.ArrayContainer` to
    :mod:`numpy` using the provided :class:`~arraycontext.ArrayContext` *actx*.

    The conversion is done using :meth:`arraycontext.ArrayContext.to_numpy`.
    """
    return rec_map_array_container(actx.to_numpy, ary)

# }}}

# vim: foldmethod=marker
