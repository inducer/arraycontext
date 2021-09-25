# mypy: disallow-untyped-defs

"""
.. currentmodule:: arraycontext

.. autofunction:: map_array_container
.. autofunction:: multimap_array_container
.. autofunction:: rec_map_array_container
.. autofunction:: rec_multimap_array_container

.. autofunction:: map_reduce_array_container
.. autofunction:: multimap_reduce_array_container
.. autofunction:: rec_map_reduce_array_container
.. autofunction:: rec_multimap_reduce_array_container

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
.. autofunction:: flatten_to_numpy
.. autofunction:: unflatten_from_numpy
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

from typing import Any, Callable, Iterable, List, Optional, Union, Tuple
from functools import update_wrapper, partial, singledispatch

import numpy as np

from arraycontext.context import ArrayContext
from arraycontext.container import (
        ContainerT, ArrayOrContainerT, is_array_container,
        serialize_container, deserialize_container)


# {{{ array container traversal helpers

def _map_array_container_impl(
        f: Callable[[Any], Any],
        ary: ArrayOrContainerT, *,
        leaf_cls: Optional[type] = None,
        recursive: bool = False) -> ArrayOrContainerT:
    """Helper for :func:`rec_map_array_container`.

    :param leaf_cls: class on which we call *f* directly. This is mostly
        useful in the recursive setting, where it can stop the recursion on
        specific container classes. By default, the recursion is stopped when
        a non-:class:`ArrayContainer` class is encountered.
    """
    def rec(_ary: ArrayOrContainerT) -> ArrayOrContainerT:
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


def _multimap_array_container_impl(
        f: Callable[..., Any],
        *args: Any,
        reduce_func: Callable[[ContainerT, Iterable[Tuple[Any, Any]]], Any] = None,
        leaf_cls: Optional[type] = None,
        recursive: bool = False) -> ArrayOrContainerT:
    """Helper for :func:`rec_multimap_array_container`.

    :param leaf_cls: class on which we call *f* directly. This is mostly
        useful in the recursive setting, where it can stop the recursion on
        specific container classes. By default, the recursion is stopped when
        a non-:class:`ArrayContainer` class is encountered.
    """
    def rec(*_args: Any) -> Any:
        template_ary = _args[container_indices[0]]
        if (type(template_ary) is leaf_cls
                or not is_array_container(template_ary)):
            return f(*_args)

        assert all(
                type(_args[i]) is type(template_ary) for i in container_indices[1:]
                ), f"expected type '{type(template_ary).__name__}'"

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

            result.append((key, frec(*new_args)))       # type: ignore[operator]

        return process_container(template_ary, result)     # type: ignore[operator]

    container_indices: List[int] = [
            i for i, arg in enumerate(args)
            if is_array_container(arg) and type(arg) is not leaf_cls]

    if not container_indices:
        return f(*args)

    if len(container_indices) == 1 and reduce_func is None:
        # NOTE: if we just have one ArrayContainer in args, passing it through
        # _map_array_container_impl should be faster
        def wrapper(ary: ArrayOrContainerT) -> ArrayOrContainerT:
            new_args = list(args)
            new_args[container_indices[0]] = ary
            return f(*new_args)

        update_wrapper(wrapper, f)
        template_ary: ContainerT = args[container_indices[0]]
        return _map_array_container_impl(
                wrapper, template_ary,
                leaf_cls=leaf_cls, recursive=recursive)

    process_container = deserialize_container if reduce_func is None else reduce_func
    frec = rec if recursive else f

    return rec(*args)

# }}}


# {{{ array container traversal

def map_array_container(
        f: Callable[[Any], Any],
        ary: ArrayOrContainerT) -> ArrayOrContainerT:
    r"""Applies *f* to all components of an :class:`ArrayContainer`.

    Works similarly to :func:`~pytools.obj_array.obj_array_vectorize`, but
    on arbitrary containers.

    For a recursive version, see :func:`rec_map_array_container`.

    :param ary: a (potentially nested) structure of :class:`ArrayContainer`\ s,
        or an instance of a base array type.
    """
    try:
        iterable = serialize_container(ary)
    except TypeError:
        return f(ary)
    else:
        return deserialize_container(ary, [
            (key, f(subary)) for key, subary in iterable
            ])


def multimap_array_container(f: Callable[..., Any], *args: Any) -> Any:
    r"""Applies *f* to the components of multiple :class:`ArrayContainer`\ s.

    Works similarly to :func:`~pytools.obj_array.obj_array_vectorize_n_args`,
    but on arbitrary containers. The containers must all have the same type,
    which will also be the return type.

    For a recursive version, see :func:`rec_multimap_array_container`.

    :param args: all :class:`ArrayContainer` arguments must be of the same
        type and with the same structure (same number of components, etc.).
    """
    return _multimap_array_container_impl(f, *args, recursive=False)


def rec_map_array_container(
        f: Callable[[Any], Any],
        ary: ArrayOrContainerT) -> ArrayOrContainerT:
    r"""Applies *f* recursively to an :class:`ArrayContainer`.

    For a non-recursive version see :func:`map_array_container`.

    :param ary: a (potentially nested) structure of :class:`ArrayContainer`\ s,
        or an instance of a base array type.
    """
    return _map_array_container_impl(f, ary, recursive=True)


def mapped_over_array_containers(
        f: Callable[[Any], Any]) -> Callable[[ArrayOrContainerT], ArrayOrContainerT]:
    """Decorator around :func:`rec_map_array_container`."""
    wrapper = partial(rec_map_array_container, f)
    update_wrapper(wrapper, f)
    return wrapper


def rec_multimap_array_container(f: Callable[..., Any], *args: Any) -> Any:
    r"""Applies *f* recursively to multiple :class:`ArrayContainer`\ s.

    For a non-recursive version see :func:`multimap_array_container`.

    :param args: all :class:`ArrayContainer` arguments must be of the same
        type and with the same structure (same number of components, etc.).
    """
    return _multimap_array_container_impl(f, *args, recursive=True)


def multimapped_over_array_containers(
        f: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator around :func:`rec_multimap_array_container`."""
    # can't use functools.partial, because its result is insufficiently
    # function-y to be used as a method definition.
    def wrapper(*args: Any) -> Any:
        return rec_multimap_array_container(f, *args)

    update_wrapper(wrapper, f)
    return wrapper

# }}}


# {{{ keyed array container traversal

def keyed_map_array_container(f: Callable[[Any, Any], Any],
                              ary: ArrayOrContainerT) -> ArrayOrContainerT:
    r"""Applies *f* to all components of an :class:`ArrayContainer`.

    Works similarly to :func:`map_array_container`, but *f* also takes an
    identifier of the array in the container *ary*.

    For a recursive version, see :func:`rec_keyed_map_array_container`.

    :param ary: a (potentially nested) structure of :class:`ArrayContainer`\ s,
        or an instance of a base array type.
    """
    try:
        iterable = serialize_container(ary)
    except TypeError:
        raise ValueError(
                f"Non-array container type has no key: {type(ary).__name__}")
    else:
        return deserialize_container(ary, [
            (key, f(key, subary)) for key, subary in iterable
            ])


def rec_keyed_map_array_container(f: Callable[[Tuple[Any, ...], Any], Any],
                                  ary: ArrayOrContainerT) -> ArrayOrContainerT:
    """
    Works similarly to :func:`rec_map_array_container`, except that *f* also
    takes in a traversal path to the leaf array. The traversal path argument is
    passed in as a tuple of identifiers of the arrays traversed before reaching
    the current array.
    """

    def rec(keys: Tuple[Union[str, int], ...],
            _ary: ArrayOrContainerT) -> ArrayOrContainerT:
        try:
            iterable = serialize_container(_ary)
        except TypeError:
            return f(keys, _ary)
        else:
            return deserialize_container(_ary, [
                (key, rec(keys + (key,), subary)) for key, subary in iterable
                ])

    return rec((), ary)

# }}}


# {{{ array container reductions

def map_reduce_array_container(
        reduce_func: Callable[[Iterable[Any]], Any],
        map_func: Callable[[Any], Any],
        ary: ArrayOrContainerT) -> Any:
    """Perform a map-reduce over array containers.

    :param reduce_func: callable used to reduce over the components of *ary*
        if *ary* is an :class:`~arraycontext.ArrayContainer`. The callable
        should be associative, as for :func:`rec_map_reduce_array_container`.
    :param map_func: callable used to map a single array of type
        :class:`arraycontext.ArrayContext.array_types`. Returns an array of the
        same type or a scalar.
    """
    try:
        iterable = serialize_container(ary)
    except TypeError:
        return map_func(ary)
    else:
        return reduce_func([
            map_func(subary) for _, subary in iterable
            ])


def multimap_reduce_array_container(
        reduce_func: Callable[[Iterable[Any]], Any],
        map_func: Callable[..., Any],
        *args: Any) -> Any:
    r"""Perform a map-reduce over multiple array containers.

    :param reduce_func: callable used to reduce over the components of any
        :class:`~arraycontext.ArrayContainer`\ s in *\*args*. The callable
        should be associative, as for :func:`rec_map_reduce_array_container`.
    :param map_func: callable used to map a single array of type
        :class:`arraycontext.ArrayContext.array_types`. Returns an array of the
        same type or a scalar.
    """
    # NOTE: this wrapper matches the signature of `deserialize_container`
    # to make plugging into `_multimap_array_container_impl` easier
    def _reduce_wrapper(ary: ContainerT, iterable: Iterable[Tuple[Any, Any]]) -> Any:
        return reduce_func([subary for _, subary in iterable])

    return _multimap_array_container_impl(
        map_func, *args,
        reduce_func=_reduce_wrapper, leaf_cls=None, recursive=False)


def rec_map_reduce_array_container(
        reduce_func: Callable[[Iterable[Any]], Any],
        map_func: Callable[[Any], Any],
        ary: ArrayOrContainerT) -> Any:
    """Perform a map-reduce over array containers recursively.

    :param reduce_func: callable used to reduce over the components of *ary*
        (and those of its sub-containers) if *ary* is a
        :class:`~arraycontext.ArrayContainer`. Must be associative.
    :param map_func: callable used to map a single array of type
        :class:`arraycontext.ArrayContext.array_types`. Returns an array of the
        same type or a scalar.

    .. note::

        The traversal order is unspecified. *reduce_func* must be associative in
        order to guarantee a sensible result. This is because *reduce_func* may be
        called on subsets of the component arrays, and then again (potentially
        multiple times) on the results. As an example, consider a container made up
        of two sub-containers, *subcontainer0* and *subcontainer1*, that each
        contain two component arrays, *array0* and *array1*. The same result must be
        computed whether traversing recursively::

            reduce_func([
                reduce_func([
                    map_func(subcontainer0.array0),
                    map_func(subcontainer0.array1)]),
                reduce_func([
                    map_func(subcontainer1.array0),
                    map_func(subcontainer1.array1)])])

        reducing all of the arrays at once::

            reduce_func([
                map_func(subcontainer0.array0),
                map_func(subcontainer0.array1),
                map_func(subcontainer1.array0),
                map_func(subcontainer1.array1)])

        or any other such traversal.
    """
    def rec(_ary: ArrayOrContainerT) -> ArrayOrContainerT:
        try:
            iterable = serialize_container(_ary)
        except TypeError:
            return map_func(_ary)
        else:
            return reduce_func([
                rec(subary) for _, subary in iterable
                ])

    return rec(ary)


def rec_multimap_reduce_array_container(
        reduce_func: Callable[[Iterable[Any]], Any],
        map_func: Callable[..., Any],
        *args: Any) -> Any:
    r"""Perform a map-reduce over multiple array containers recursively.

    :param reduce_func: callable used to reduce over the components of any
        :class:`~arraycontext.ArrayContainer`\ s in *\*args* (and those of their
        sub-containers). Must be associative.
    :param map_func: callable used to map a single array of type
        :class:`arraycontext.ArrayContext.array_types`. Returns an array of the
        same type or a scalar.

    .. note::

        The traversal order is unspecified. *reduce_func* must be associative in
        order to guarantee a sensible result. See
        :func:`rec_map_reduce_array_container` for additional details.
    """
    # NOTE: this wrapper matches the signature of `deserialize_container`
    # to make plugging into `_multimap_array_container_impl` easier
    def _reduce_wrapper(ary: ContainerT, iterable: Iterable[Tuple[Any, Any]]) -> Any:
        return reduce_func([subary for _, subary in iterable])

    return _multimap_array_container_impl(
        map_func, *args,
        reduce_func=_reduce_wrapper, leaf_cls=None, recursive=True)

# }}}


# {{{ freeze/thaw

@singledispatch
def freeze(
        ary: ArrayOrContainerT,
        actx: Optional[ArrayContext] = None) -> ArrayOrContainerT:
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
def thaw(ary: ArrayOrContainerT, actx: ArrayContext) -> ArrayOrContainerT:
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
    try:
        iterable = serialize_container(ary)
    except TypeError:
        return actx.thaw(ary)
    else:
        return deserialize_container(ary, [
            (key, thaw(subary, actx)) for key, subary in iterable
            ])

# }}}


# {{{ numpy conversion

def from_numpy(ary: Any, actx: ArrayContext) -> Any:
    """Convert all :mod:`numpy` arrays in the :class:`~arraycontext.ArrayContainer`
    to the base array type of :class:`~arraycontext.ArrayContext`.

    The conversion is done using :meth:`arraycontext.ArrayContext.from_numpy`.
    """
    def _from_numpy(subary: Any) -> Any:
        if isinstance(subary, np.ndarray) and subary.dtype != "O":
            return actx.from_numpy(subary)
        elif is_array_container(subary):
            return map_array_container(_from_numpy, subary)
        else:
            raise TypeError(f"unrecognized array type: '{type(subary).__name__}'")

    return _from_numpy(ary)


def to_numpy(ary: Any, actx: ArrayContext) -> Any:
    """Convert all arrays in the :class:`~arraycontext.ArrayContainer` to
    :mod:`numpy` using the provided :class:`~arraycontext.ArrayContext` *actx*.

    The conversion is done using :meth:`arraycontext.ArrayContext.to_numpy`.
    """
    return rec_map_array_container(actx.to_numpy, ary)


def flatten_to_numpy(ary: ArrayOrContainerT, actx: ArrayContext) -> np.ndarray:
    """Convert all arrays in the :class:`~arraycontext.ArrayContainer`
    to host :mod:`numpy` arrays, flatten them using :func:`~numpy.ravel`
    and concatenate them into a single :class:`~numpy.ndarray`.

    The order in which the individual leaf arrays appear in the final array is
    dependent on the order given by :func:`~arraycontext.serialize_container`.
    """
    def _flatten_to_numpy(subary: ArrayOrContainerT) -> None:
        try:
            iterable = serialize_container(subary)
        except TypeError:
            result.append(actx.to_numpy(subary).ravel())
        else:
            for _, isubary in iterable:
                _flatten_to_numpy(isubary)

    result: List[np.ndarray] = []
    _flatten_to_numpy(ary)

    return np.concatenate(result)


def unflatten_from_numpy(
        template: ArrayOrContainerT, ary: np.ndarray,
        actx: ArrayContext) -> ArrayOrContainerT:
    """Unflatten an :class:`~numpy.ndarray` produced by :func:`flatten_to_numpy`
    back into an :class:`~arraycontext.ArrayContainer`.

    The order and sizes of each slice into *ary* are determined by the
    array container *template*.
    """
    # NOTE: https://github.com/python/mypy/issues/7057
    offset = 0

    def _unflatten_from_numpy(subary: ArrayOrContainerT) -> ArrayOrContainerT:
        nonlocal offset

        try:
            iterable = serialize_container(subary)
        except TypeError:
            # NOTE: the max is needed to handle device scalars with size == 0
            offset += max(1, subary.size)
            if offset > ary.size:
                raise ValueError("'template' and 'ary' sizes do not match")

            # FIXME: subary can be F-contiguous and ary will always be C-contiguous
            return actx.from_numpy(
                    ary[offset - subary.size:offset]
                    .astype(subary.dtype, copy=False)
                    .reshape(subary.shape)
                    )
        else:
            return deserialize_container(subary, [
                (key, _unflatten_from_numpy(isubary)) for key, isubary in iterable
                ])

    if ary.ndim != 1:
        raise ValueError(
                "only one dimensional arrays can be unflattened: "
                f"'ary' has shape {ary.shape}")

    return _unflatten_from_numpy(template)

# }}}

# vim: foldmethod=marker
