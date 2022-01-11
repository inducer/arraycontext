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

Flattening and unflattening
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: flatten
.. autofunction:: unflatten

Numpy conversion
~~~~~~~~~~~~~~~~
.. autofunction:: from_numpy
.. autofunction:: to_numpy

Algebraic operations
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: outer
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

from numbers import Number
from typing import Any, Callable, Iterable, List, Optional, Union, Tuple
from functools import update_wrapper, partial, singledispatch

import numpy as np

from arraycontext.context import ArrayContext, DeviceArray
from arraycontext.container import (
        ContainerT, ArrayOrContainerT, NotAnArrayContainerError,
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

        try:
            iterable = serialize_container(_ary)
        except NotAnArrayContainerError:
            return f(_ary)
        else:
            return deserialize_container(_ary, [
                (key, frec(subary)) for key, subary in iterable
                ])

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

    # {{{ recursive traversal

    def rec(*_args: Any) -> Any:
        template_ary = _args[container_indices[0]]
        if type(template_ary) is leaf_cls:
            return f(*_args)

        try:
            iterable_template = serialize_container(template_ary)
        except NotAnArrayContainerError:
            return f(*_args)
        else:
            pass

        assert all(
                type(_args[i]) is type(template_ary) for i in container_indices[1:]
                ), f"expected type '{type(template_ary).__name__}'"

        result = []
        new_args = list(_args)

        for subarys in zip(
                iterable_template,
                *[serialize_container(_args[i]) for i in container_indices[1:]]
                ):
            key = None
            for i, (subkey, subary) in zip(container_indices, subarys):
                if key is None:
                    key = subkey
                else:
                    assert key == subkey

                new_args[i] = subary

            result.append((key, frec(*new_args)))       # type: ignore[operator]

        return process_container(template_ary, result)     # type: ignore[operator]

    # }}}

    # {{{ find all containers in the argument list

    container_indices: List[int] = []

    for i, arg in enumerate(args):
        if type(arg) is leaf_cls:
            continue

        try:
            # FIXME: this will serialize again once `rec` is called, which is
            # not great, but it doesn't seem like there's a good way to avoid it
            _ = serialize_container(arg)
        except NotAnArrayContainerError:
            pass
        else:
            container_indices.append(i)

    # }}}

    # {{{ #containers == 0 => call `f`

    if not container_indices:
        return f(*args)

    # }}}

    # {{{ #containers == 1 => call `map_array_container`

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

    # }}}

    # {{{ #containers > 1 => call `rec`

    process_container = deserialize_container if reduce_func is None else reduce_func
    frec = rec if recursive else f

    # }}}

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
    except NotAnArrayContainerError:
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
        ary: ArrayOrContainerT,
        leaf_class: Optional[type] = None) -> ArrayOrContainerT:
    r"""Applies *f* recursively to an :class:`ArrayContainer`.

    For a non-recursive version see :func:`map_array_container`.

    :param ary: a (potentially nested) structure of :class:`ArrayContainer`\ s,
        or an instance of a base array type.
    """
    return _map_array_container_impl(f, ary, leaf_cls=leaf_class, recursive=True)


def mapped_over_array_containers(
        f: Optional[Callable[[Any], Any]] = None,
        leaf_class: Optional[type] = None) -> Union[
            Callable[[ArrayOrContainerT], ArrayOrContainerT],
            Callable[
                [Callable[[Any], Any]],
                Callable[[ArrayOrContainerT], ArrayOrContainerT]]]:
    """Decorator around :func:`rec_map_array_container`."""
    def decorator(g: Callable[[Any], Any]) -> Callable[
            [ArrayOrContainerT], ArrayOrContainerT]:
        wrapper = partial(rec_map_array_container, g, leaf_class=leaf_class)
        update_wrapper(wrapper, g)
        return wrapper
    if f is not None:
        return decorator(f)
    else:
        return decorator


def rec_multimap_array_container(
        f: Callable[..., Any],
        *args: Any,
        leaf_class: Optional[type] = None) -> Any:
    r"""Applies *f* recursively to multiple :class:`ArrayContainer`\ s.

    For a non-recursive version see :func:`multimap_array_container`.

    :param args: all :class:`ArrayContainer` arguments must be of the same
        type and with the same structure (same number of components, etc.).
    """
    return _multimap_array_container_impl(
        f, *args, leaf_cls=leaf_class, recursive=True)


def multimapped_over_array_containers(
        f: Optional[Callable[..., Any]] = None,
        leaf_class: Optional[type] = None) -> Union[
            Callable[..., Any],
            Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """Decorator around :func:`rec_multimap_array_container`."""
    def decorator(g: Callable[..., Any]) -> Callable[..., Any]:
        # can't use functools.partial, because its result is insufficiently
        # function-y to be used as a method definition.
        def wrapper(*args: Any) -> Any:
            return rec_multimap_array_container(g, *args, leaf_class=leaf_class)
        update_wrapper(wrapper, g)
        return wrapper
    if f is not None:
        return decorator(f)
    else:
        return decorator


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
    except NotAnArrayContainerError:
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
        except NotAnArrayContainerError:
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
        ary: ArrayOrContainerT) -> "DeviceArray":
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
    except NotAnArrayContainerError:
        return map_func(ary)
    else:
        return reduce_func([
            map_func(subary) for _, subary in iterable
            ])


def multimap_reduce_array_container(
        reduce_func: Callable[[Iterable[Any]], Any],
        map_func: Callable[..., Any],
        *args: Any) -> "DeviceArray":
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
        ary: ArrayOrContainerT,
        leaf_class: Optional[type] = None) -> "DeviceArray":
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
        if type(_ary) is leaf_class:
            return map_func(_ary)
        else:
            try:
                iterable = serialize_container(_ary)
            except NotAnArrayContainerError:
                return map_func(_ary)
            else:
                return reduce_func([
                    rec(subary) for _, subary in iterable
                    ])

    return rec(ary)


def rec_multimap_reduce_array_container(
        reduce_func: Callable[[Iterable[Any]], Any],
        map_func: Callable[..., Any],
        *args: Any,
        leaf_class: Optional[type] = None) -> "DeviceArray":
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
        reduce_func=_reduce_wrapper, leaf_cls=leaf_class, recursive=True)

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
    try:
        iterable = serialize_container(ary)
    except NotAnArrayContainerError:
        if actx is None:
            raise TypeError(
                    f"cannot freeze arrays of type {type(ary).__name__} "
                    "when actx is not supplied. Try calling actx.freeze "
                    "directly or supplying an array context")
        else:
            return actx.freeze(ary)
    else:
        return deserialize_container(ary, [
            (key, freeze(subary, actx=actx)) for key, subary in iterable
            ])


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
    except NotAnArrayContainerError:
        return actx.thaw(ary)
    else:
        return deserialize_container(ary, [
            (key, thaw(subary, actx)) for key, subary in iterable
            ])

# }}}


# {{{ flatten / unflatten

def flatten(ary: ArrayOrContainerT, actx: ArrayContext) -> Any:
    """Convert all arrays in the :class:`~arraycontext.ArrayContainer`
    into single flat array of a type :attr:`arraycontext.ArrayContext.array_types`.

    The operation requires :attr:`arraycontext.ArrayContext.np` to have
    ``ravel`` and ``concatenate`` methods implemented. The order in which the
    individual leaf arrays appear in the final array is dependent on the order
    given by :func:`~arraycontext.serialize_container`.
    """
    common_dtype = None
    result: List[Any] = []

    def _flatten(subary: ArrayOrContainerT) -> None:
        nonlocal common_dtype

        try:
            iterable = serialize_container(subary)
        except NotAnArrayContainerError:
            if common_dtype is None:
                common_dtype = subary.dtype

            if subary.dtype != common_dtype:
                raise ValueError("arrays in container have different dtypes: "
                        f"got {subary.dtype}, expected {common_dtype}")

            try:
                flat_subary = actx.np.ravel(subary, order="C")
            except ValueError as exc:
                # NOTE: we can't do much if the array context fails to ravel,
                # since it is the one responsible for the actual memory layout
                if hasattr(subary, "strides"):
                    strides_msg = f" and strides {subary.strides}"
                else:
                    strides_msg = ""

                raise NotImplementedError(
                        f"'{type(actx).__name__}.np.ravel' failed to reshape "
                        f"an array with shape {subary.shape}{strides_msg}. "
                        "This functionality needs to be implemented by the "
                        "array context.") from exc

            result.append(flat_subary)
        else:
            for _, isubary in iterable:
                _flatten(isubary)

    _flatten(ary)

    return actx.np.concatenate(result)


def unflatten(
        template: ArrayOrContainerT, ary: Any,
        actx: ArrayContext, *,
        strict: bool = True) -> ArrayOrContainerT:
    """Unflatten an array *ary* produced by :func:`flatten` back into an
    :class:`~arraycontext.ArrayContainer`.

    The order and sizes of each slice into *ary* are determined by the
    array container *template*.

    :arg strict: if *True* additional :class:`~numpy.dtype` and stride
        checking is performed on the unflattened array. Otherwise, these
        checks are skipped.
    """
    # NOTE: https://github.com/python/mypy/issues/7057
    offset = 0
    common_dtype = None

    def _unflatten(template_subary: ArrayOrContainerT) -> ArrayOrContainerT:
        nonlocal offset, common_dtype

        try:
            iterable = serialize_container(template_subary)
        except NotAnArrayContainerError:
            # {{{ validate subary

            if (offset + template_subary.size) > ary.size:
                raise ValueError("'template' and 'ary' sizes do not match: "
                    "'template' is too large")

            if strict:
                if template_subary.dtype != ary.dtype:
                    raise ValueError("'template' dtype does not match 'ary': "
                            f"got {template_subary.dtype}, expected {ary.dtype}")
            else:
                # NOTE: still require that *template* has a uniform dtype
                if common_dtype is None:
                    common_dtype = template_subary.dtype
                else:
                    if common_dtype != template_subary.dtype:
                        raise ValueError("arrays in 'template' have different "
                                f"dtypes: got {template_subary.dtype}, but "
                                f"expected {common_dtype}.")

            # }}}

            # {{{ reshape

            flat_subary = ary[offset:offset + template_subary.size]
            try:
                subary = actx.np.reshape(flat_subary,
                        template_subary.shape, order="C")
            except ValueError as exc:
                # NOTE: we can't do much if the array context fails to reshape,
                # since it is the one responsible for the actual memory layout
                raise NotImplementedError(
                        f"'{type(actx).__name__}.np.reshape' failed to reshape "
                        f"the flat array into shape {template_subary.shape}. "
                        "This functionality needs to be implemented by the "
                        "array context.") from exc

            # }}}

            # {{{ check strides

            if strict and hasattr(template_subary, "strides"):
                if template_subary.strides != subary.strides:
                    raise ValueError(
                            f"strides do not match template: got {subary.strides}, "
                            f"expected {template_subary.strides}")

            # }}}

            offset += template_subary.size
            return subary
        else:
            return deserialize_container(template_subary, [
                (key, _unflatten(isubary)) for key, isubary in iterable
                ])

    if not isinstance(ary, actx.array_types):
        raise TypeError("'ary' does not have a type supported by the provided "
                f"array context: got '{type(ary).__name__}', expected one of "
                f"{actx.array_types}")

    if ary.ndim != 1:
        raise ValueError(
                "only one dimensional arrays can be unflattened: "
                f"'ary' has shape {ary.shape}")

    result = _unflatten(template)
    if offset != ary.size:
        raise ValueError("'template' and 'ary' sizes do not match: "
            "'ary' is too large")

    return result

# }}}


# {{{ numpy conversion

def from_numpy(
        ary: Union[np.ndarray, np.generic, Number],
        actx: ArrayContext) -> ArrayOrContainerT:
    """Convert all :mod:`numpy` arrays in the :class:`~arraycontext.ArrayContainer`
    to the base array type of :class:`~arraycontext.ArrayContext`.

    The conversion is done using :meth:`arraycontext.ArrayContext.from_numpy`.
    """
    def _from_numpy_with_check(subary: Union[np.ndarray, np.generic, Number]) \
            -> ArrayOrContainerT:
        if isinstance(subary, np.ndarray) or np.isscalar(subary):
            return actx.from_numpy(subary)
        else:
            raise TypeError(f"array is not an ndarray: '{type(subary).__name__}'")

    return rec_map_array_container(_from_numpy_with_check, ary)


def to_numpy(ary: ArrayOrContainerT, actx: ArrayContext) -> Any:
    """Convert all arrays in the :class:`~arraycontext.ArrayContainer` to
    :mod:`numpy` using the provided :class:`~arraycontext.ArrayContext` *actx*.

    The conversion is done using :meth:`arraycontext.ArrayContext.to_numpy`.
    """
    def _to_numpy_with_check(subary: Any) -> Any:
        if isinstance(subary, actx.array_types) or np.isscalar(subary):
            return actx.to_numpy(subary)
        else:
            raise TypeError(
                    f"array of type '{type(subary).__name__}' not in "
                    f"supported types {actx.array_types}")

    return rec_map_array_container(_to_numpy_with_check, ary)

# }}}


# {{{ algebraic operations

def outer(a: Any, b: Any) -> Any:
    """
    Compute the outer product of *a* and *b* while allowing either of them
    to be an :class:`ArrayContainer`.

    Tweaks the behavior of :func:`numpy.outer` to return a lower-dimensional
    object if either/both of *a* and *b* are scalars (whereas :func:`numpy.outer`
    always returns a matrix). Here the definition of "scalar" includes
    all non-array-container types and any scalar-like array container types
    (including non-object numpy arrays).

    If *a* and *b* are both array containers, the result will have the same type
    as *a*. If both are array containers and neither is an object array, they must
    have the same type.
    """

    def treat_as_scalar(x: Any) -> bool:
        try:
            serialize_container(x)
        except NotAnArrayContainerError:
            return True
        else:
            return (
                not isinstance(x, np.ndarray)
                # This condition is whether "ndarrays should broadcast inside x".
                and np.ndarray not in x.__class__._outer_bcast_types)

    if treat_as_scalar(a) or treat_as_scalar(b):
        return a*b
    # After this point, "isinstance(o, ndarray)" means o is an object array.
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.outer(a, b)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return map_array_container(lambda x: outer(x, b), a)
    else:
        if type(a) != type(b):
            raise TypeError(
                "both arguments must have the same type if they are both "
                "non-object-array array containers.")
        return multimap_array_container(lambda x, y: outer(x, y), a, b)

# }}}

# vim: foldmethod=marker
