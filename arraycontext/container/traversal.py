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

.. autofunction:: stringify_array_container_tree

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
.. autofunction:: flat_size_and_dtype

Numpy conversion
~~~~~~~~~~~~~~~~
.. autofunction:: from_numpy
.. autofunction:: to_numpy

Algebraic operations
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: outer
"""

from __future__ import annotations

from arraycontext.container.arithmetic import NumpyObjectArray


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

from functools import partial, singledispatch, update_wrapper
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, cast
from warnings import warn

import numpy as np

from arraycontext.container import (
    ArrayContainer,
    NotAnArrayContainerError,
    SerializationKey,
    deserialize_container,
    get_container_context_recursively_opt,
    serialize_container,
)
from arraycontext.context import (
    Array,
    ArrayContext,
    ArrayOrContainer,
    ArrayOrContainerOrScalar,
    ArrayOrContainerT,
    ArrayT,
    ScalarLike,
)


# {{{ array container traversal helpers

def _map_array_container_impl(
        f: Callable[[ArrayOrContainer], ArrayOrContainer],
        ary: ArrayOrContainer, *,
        leaf_cls: Optional[type] = None,
        recursive: bool = False) -> ArrayOrContainer:
    """Helper for :func:`rec_map_array_container`.

    :param leaf_cls: class on which we call *f* directly. This is mostly
        useful in the recursive setting, where it can stop the recursion on
        specific container classes. By default, the recursion is stopped when
        a non-:class:`ArrayContainer` class is encountered.
    """
    def rec(_ary: ArrayOrContainer) -> ArrayOrContainer:
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
        reduce_func: Optional[Callable[
            [ArrayContainer, Iterable[Tuple[Any, Any]]], Any]] = None,
        leaf_cls: Optional[type] = None,
        recursive: bool = False) -> ArrayOrContainer:
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
        template_ary: ArrayContainer = args[container_indices[0]]
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

def stringify_array_container_tree(ary: ArrayOrContainer) -> str:
    """
    :returns: a string for an ASCII tree representation of the array container,
        similar to `asciitree <https://github.com/mbr/asciitree>`__.
    """
    def rec(lines: List[str], ary_: ArrayOrContainerT, level: int) -> None:
        try:
            iterable = serialize_container(ary_)
        except NotAnArrayContainerError:
            pass
        else:
            for key, subary in iterable:
                key = f"{key} ({type(subary).__name__})"
                if level == 0:
                    indent = ""
                else:
                    indent = f" |  {' ' * 4 * (level - 1)}"

                lines.append(f"{indent} +-- {key}")
                rec(lines, subary, level + 1)

    lines = [f"root ({type(ary).__name__})"]
    rec(lines, ary, 0)

    return "\n".join(lines)


def map_array_container(
        f: Callable[[Any], Any],
        ary: ArrayOrContainer) -> ArrayOrContainer:
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
        ary: ArrayOrContainer,
        leaf_class: Optional[type] = None) -> ArrayOrContainer:
    r"""Applies *f* recursively to an :class:`ArrayContainer`.

    For a non-recursive version see :func:`map_array_container`.

    :param ary: a (potentially nested) structure of :class:`ArrayContainer`\ s,
        or an instance of a base array type.
    """
    return _map_array_container_impl(f, ary, leaf_cls=leaf_class, recursive=True)


def mapped_over_array_containers(
        f: Optional[Callable[[ArrayOrContainer], ArrayOrContainer]] = None,
        leaf_class: Optional[type] = None) -> Union[
            Callable[[ArrayOrContainer], ArrayOrContainer],
            Callable[
                [Callable[[Any], Any]],
                Callable[[ArrayOrContainer], ArrayOrContainer]]]:
    """Decorator around :func:`rec_map_array_container`."""
    def decorator(g: Callable[[ArrayOrContainer], ArrayOrContainer]) -> Callable[
            [ArrayOrContainer], ArrayOrContainer]:
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

def keyed_map_array_container(
        f: Callable[
            [SerializationKey, ArrayOrContainer],
            ArrayOrContainer],
        ary: ArrayOrContainer) -> ArrayOrContainer:
    r"""Applies *f* to all components of an :class:`ArrayContainer`.

    Works similarly to :func:`map_array_container`, but *f* also takes an
    identifier of the array in the container *ary*.

    For a recursive version, see :func:`rec_keyed_map_array_container`.

    :param ary: a (potentially nested) structure of :class:`ArrayContainer`\ s,
        or an instance of a base array type.
    """
    try:
        iterable = serialize_container(ary)
    except NotAnArrayContainerError as err:
        raise ValueError(
                f"Non-array container type has no key: {type(ary).__name__}") from err
    else:
        return deserialize_container(ary, [
            (key, f(key, subary)) for key, subary in iterable
            ])


def rec_keyed_map_array_container(
        f: Callable[[Tuple[SerializationKey, ...], ArrayT], ArrayT],
        ary: ArrayOrContainer) -> ArrayOrContainer:
    """
    Works similarly to :func:`rec_map_array_container`, except that *f* also
    takes in a traversal path to the leaf array. The traversal path argument is
    passed in as a tuple of identifiers of the arrays traversed before reaching
    the current array.
    """

    def rec(keys: Tuple[SerializationKey, ...],
            _ary: ArrayOrContainerT) -> ArrayOrContainerT:
        try:
            iterable = serialize_container(_ary)
        except NotAnArrayContainerError:
            return cast(ArrayOrContainerT, f(keys, cast(ArrayT, _ary)))
        else:
            return deserialize_container(_ary, [
                (key, rec((*keys, key), subary)) for key, subary in iterable
                ])

    return rec((), ary)

# }}}


# {{{ array container reductions

def map_reduce_array_container(
        reduce_func: Callable[[Iterable[Any]], Any],
        map_func: Callable[[Any], Any],
        ary: ArrayOrContainerT) -> Array:
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
        *args: Any) -> ArrayOrContainer:
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
    def _reduce_wrapper(
            ary: ArrayContainer, iterable: Iterable[Tuple[Any, Any]]
            ) -> Array:
        return reduce_func([subary for _, subary in iterable])

    return _multimap_array_container_impl(
        map_func, *args,
        reduce_func=_reduce_wrapper, leaf_cls=None, recursive=False)


def rec_map_reduce_array_container(
        reduce_func: Callable[[Iterable[Any]], Any],
        map_func: Callable[[Any], Any],
        ary: ArrayOrContainer,
        leaf_class: Optional[type] = None) -> ArrayOrContainer:
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
        leaf_class: Optional[type] = None) -> ArrayOrContainer:
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
    def _reduce_wrapper(
            ary: ArrayContainer, iterable: Iterable[Tuple[Any, Any]]) -> Any:
        return reduce_func([subary for _, subary in iterable])

    return _multimap_array_container_impl(
        map_func, *args,
        reduce_func=_reduce_wrapper, leaf_cls=leaf_class, recursive=True)

# }}}


# {{{ freeze/thaw

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

    if actx is None:
        warn("Calling freeze(ary) without specifying actx is deprecated, explicitly"
             " call actx.freeze(ary) instead. This will stop working in 2023.",
             DeprecationWarning, stacklevel=2)

        actx = get_container_context_recursively_opt(ary)
    else:
        warn("Calling freeze(ary, actx) is deprecated, call actx.freeze(ary)"
             " instead. This will stop working in 2023.",
             DeprecationWarning, stacklevel=2)

        if __debug__:
            rec_actx = get_container_context_recursively_opt(ary)
            if (rec_actx is not None) and (rec_actx is not actx):
                raise ValueError("Supplied array context does not agree with"
                                 " the one obtained by traversing 'ary'.")

    if actx is None:
        raise TypeError(
                f"cannot freeze arrays of type {type(ary).__name__} "
                "when actx is not supplied. Try calling actx.freeze "
                "directly or supplying an array context")

    return actx.freeze(ary)


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
    warn("Calling thaw(ary, actx) is deprecated, call actx.thaw(ary) instead."
         " This will stop working in 2023.",
         DeprecationWarning, stacklevel=2)

    if __debug__:
        rec_actx = get_container_context_recursively_opt(ary)
        if rec_actx is not None:
            raise ValueError("cannot thaw a container that already has an array"
                             " context.")

    return actx.thaw(ary)

# }}}


# {{{ with_array_context

@singledispatch
def with_array_context(ary: ArrayOrContainerT,
                       actx: Optional[ArrayContext]) -> ArrayOrContainerT:
    """
    Recursively associates *actx* to all the components of *ary*.

    Array container types may use :func:`functools.singledispatch` ``.register``
    to register container-specific implementations. See `this issue
    <https://github.com/inducer/arraycontext/issues/162>`__ for discussion of
    the future of this functionality.
    """
    try:
        iterable = serialize_container(ary)
    except NotAnArrayContainerError:
        return ary
    else:
        return deserialize_container(ary, [(key, with_array_context(subary, actx))
                                           for key, subary in iterable])

# }}}


# {{{ flatten / unflatten

def flatten(
        ary: ArrayOrContainer, actx: ArrayContext, *,
        leaf_class: Optional[type] = None,
        ) -> Any:
    """Convert all arrays in the :class:`~arraycontext.ArrayContainer`
    into single flat array of a type :attr:`arraycontext.ArrayContext.array_types`.

    The operation requires :attr:`arraycontext.ArrayContext.np` to have
    ``ravel`` and ``concatenate`` methods implemented. The order in which the
    individual leaf arrays appear in the final array is dependent on the order
    given by :func:`~arraycontext.serialize_container`.

    If *leaf_class* is given, then :func:`unflatten` will not be able to recover
    the original *ary*.

    :arg leaf_class: an :class:`~arraycontext.ArrayContainer` class on which
        the recursion is stopped (subclasses are not considered). If given, only
        the entries of this type are flattened and the rest of the tree
        structure is left as is. By default, the recursion is stopped when
        a non-:class:`~arraycontext.ArrayContainer` is found, which results in
        the whole input container *ary* being flattened.
    """
    common_dtype = None

    def _flatten(subary: ArrayOrContainer) -> List[Array]:
        nonlocal common_dtype

        try:
            iterable = serialize_container(subary)
        except NotAnArrayContainerError:
            subary_c = cast(Array, subary)

            if common_dtype is None:
                common_dtype = subary_c.dtype

            if subary_c.dtype != common_dtype:
                raise ValueError("arrays in container have different dtypes: "
                        f"got {subary_c.dtype}, expected {common_dtype}") from None

            try:
                flat_subary = actx.np.ravel(subary_c, order="C")
            except ValueError as exc:
                # NOTE: we can't do much if the array context fails to ravel,
                # since it is the one responsible for the actual memory layout
                if hasattr(subary_c, "strides"):
                    strides_msg = f" and strides {subary_c.strides}"
                else:
                    strides_msg = ""

                raise NotImplementedError(
                        f"'{type(actx).__name__}.np.ravel' failed to reshape "
                        f"an array with shape {subary_c.shape}{strides_msg}. "
                        "This functionality needs to be implemented by the "
                        "array context.") from exc

            result = [flat_subary]
        else:
            result = []
            for _, isubary in iterable:
                result.extend(_flatten(isubary))

        return result

    def _flatten_without_leaf_class(subary: ArrayOrContainer) -> Any:
        result = _flatten(subary)

        if len(result) == 1:
            return result[0]
        else:
            return actx.np.concatenate(result)

    def _flatten_with_leaf_class(subary: ArrayOrContainer) -> Any:
        if type(subary) is leaf_class:
            return _flatten_without_leaf_class(subary)

        try:
            iterable = serialize_container(subary)
        except NotAnArrayContainerError:
            return subary
        else:
            return deserialize_container(subary, [
                (key, _flatten_with_leaf_class(isubary))
                for key, isubary in iterable
                ])

    if leaf_class is None:
        return _flatten_without_leaf_class(ary)
    else:
        return _flatten_with_leaf_class(ary)


def unflatten(
        template: ArrayOrContainerT, ary: Array,
        actx: ArrayContext, *,
        strict: bool = True) -> ArrayOrContainerT:
    """Unflatten an array *ary* produced by :func:`flatten` back into an
    :class:`~arraycontext.ArrayContainer`.

    The order and sizes of each slice into *ary* are determined by the
    array container *template*.

    :arg ary: a flat one-dimensional array with a size that matches the
        number of entries in *template*.
    :arg strict: if *True* additional :class:`~numpy.dtype` and stride
        checking is performed on the unflattened array. Otherwise, these
        checks are skipped.
    """
    # NOTE: https://github.com/python/mypy/issues/7057
    offset = 0
    common_dtype = None

    def _unflatten(template_subary: ArrayOrContainer) -> ArrayOrContainer:
        nonlocal offset, common_dtype

        try:
            iterable = serialize_container(template_subary)
        except NotAnArrayContainerError:
            template_subary_c = cast(Array, template_subary)

            # {{{ validate subary

            if (offset + template_subary_c.size) > ary.size:
                raise ValueError("'template' and 'ary' sizes do not match: "
                    "'template' is too large") from None

            if strict:
                if template_subary_c.dtype != ary.dtype:
                    raise ValueError("'template' dtype does not match 'ary': "
                            f"got {template_subary_c.dtype}, expected {ary.dtype}"
                        ) from None
            else:
                # NOTE: still require that *template* has a uniform dtype
                if common_dtype is None:
                    common_dtype = template_subary_c.dtype
                else:
                    if common_dtype != template_subary_c.dtype:
                        raise ValueError("arrays in 'template' have different "
                                f"dtypes: got {template_subary_c.dtype}, but "
                                f"expected {common_dtype}.") from None

            # }}}

            # {{{ reshape

            flat_subary = ary[offset:offset + template_subary_c.size]
            try:
                subary = actx.np.reshape(flat_subary,
                        template_subary_c.shape, order="C")
            except ValueError as exc:
                # NOTE: we can't do much if the array context fails to reshape,
                # since it is the one responsible for the actual memory layout
                raise NotImplementedError(
                        f"'{type(actx).__name__}.np.reshape' failed to reshape "
                        f"the flat array into shape {template_subary_c.shape}. "
                        "This functionality needs to be implemented by the "
                        "array context.") from exc

            # }}}

            # {{{ check strides

            if strict and hasattr(template_subary_c, "strides"):
                # Checking strides for 0 sized arrays is ill-defined
                # since they cannot be indexed
                if (
                    # Mypy has a point: nobody promised a .strides attribute.
                    template_subary_c.strides != subary.strides
                    and template_subary_c.size != 0
                ):
                    raise ValueError(
                            # Mypy has a point: nobody promised a .strides attribute.
                            f"strides do not match template: got {subary.strides}, "
                            f"expected {template_subary_c.strides}") from None

            # }}}

            offset += template_subary_c.size
            return subary
        else:
            return deserialize_container(template_subary, [
                        (key, _unflatten(isubary)) for key, isubary in iterable
                        ])

    if not isinstance(ary, actx.array_types):
        raise TypeError("'ary' does not have a type supported by the provided "
                f"array context: got '{type(ary).__name__}', expected one of "
                f"{actx.array_types}")

    if len(ary.shape) != 1:
        raise ValueError(
                "only one dimensional arrays can be unflattened: "
                f"'ary' has shape {ary.shape}")

    result = _unflatten(template)
    if offset != ary.size:
        raise ValueError("'template' and 'ary' sizes do not match: "
            "'ary' is too large")

    return cast(ArrayOrContainerT, result)


def flat_size_and_dtype(
        ary: ArrayOrContainer) -> Tuple[int, Optional[np.dtype[Any]]]:
    """
    :returns: a tuple ``(size, dtype)`` that would be the length and
        :class:`numpy.dtype` of the one-dimensional array returned by
        :func:`flatten`.
    """
    common_dtype = None

    def _flat_size(subary: ArrayOrContainer) -> int:
        nonlocal common_dtype

        try:
            iterable = serialize_container(subary)
        except NotAnArrayContainerError:
            subary_c = cast(Array, subary)

            if common_dtype is None:
                common_dtype = subary_c.dtype

            if subary_c.dtype != common_dtype:
                raise ValueError("arrays in container have different dtypes: "
                        f"got {subary_c.dtype}, expected {common_dtype}") from None

            return subary_c.size
        else:
            return sum(_flat_size(isubary) for _, isubary in iterable)

    size = _flat_size(ary)
    return size, common_dtype

# }}}


# {{{ numpy conversion

def from_numpy(
        ary: Union[np.ndarray, ScalarLike],
        actx: ArrayContext) -> ArrayOrContainerOrScalar:
    """Convert all :mod:`numpy` arrays in the :class:`~arraycontext.ArrayContainer`
    to the base array type of :class:`~arraycontext.ArrayContext`.

    The conversion is done using :meth:`arraycontext.ArrayContext.from_numpy`.
    """
    warn("Calling from_numpy(ary, actx) is deprecated, call actx.from_numpy(ary)"
         " instead. This will stop working in 2023.",
         DeprecationWarning, stacklevel=2)

    return actx.from_numpy(ary)


def to_numpy(ary: ArrayOrContainer, actx: ArrayContext) -> ArrayOrContainer:
    """Convert all arrays in the :class:`~arraycontext.ArrayContainer` to
    :mod:`numpy` using the provided :class:`~arraycontext.ArrayContext` *actx*.

    The conversion is done using :meth:`arraycontext.ArrayContext.to_numpy`.
    """
    warn("Calling to_numpy(ary, actx) is deprecated, call actx.to_numpy(ary)"
         " instead. This will stop working in 2023.",
         DeprecationWarning, stacklevel=2)

    return actx.to_numpy(ary)

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
                and NumpyObjectArray not in x.__class__._outer_bcast_types)

    if treat_as_scalar(a) or treat_as_scalar(b):
        return a*b
    # After this point, "isinstance(o, ndarray)" means o is an object array.
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.outer(a, b)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return map_array_container(lambda x: outer(x, b), a)
    else:
        if type(a) is not type(b):
            raise TypeError(
                "both arguments must have the same type if they are both "
                "non-object-array array containers.")
        return multimap_array_container(lambda x, y: outer(x, y), a, b)

# }}}

# vim: foldmethod=marker
