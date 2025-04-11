from __future__ import annotations


__copyright__ = "Copyright (C) 2020-21 University of Illinois Board of Trustees"

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
from dataclasses import dataclass

import numpy as np

from arraycontext import (
    ArrayContainer,
    ArrayContext,
    dataclass_array_container,
    deserialize_container,
    serialize_container,
    with_array_context,
    with_container_arithmetic,
)


# Containers live here, because in order for get_annotations to work, they must
# live somewhere importable.
# See https://docs.python.org/3.12/library/inspect.html#inspect.get_annotations


# {{{ stand-in DOFArray implementation

@with_container_arithmetic(
        bcasts_across_obj_array=True,
        bitwise=True,
        rel_comparison=True,
        _cls_has_array_context_attr=True,
        _bcast_actx_array_type=False)
class DOFArray:
    def __init__(self, actx, data):
        if not (actx is None or isinstance(actx, ArrayContext)):
            raise TypeError("actx must be of type ArrayContext")

        if not isinstance(data, tuple):
            raise TypeError("'data' argument must be a tuple")

        self.array_context = actx
        self.data = data

    # prevent numpy broadcasting
    __array_ufunc__ = None

    def __bool__(self):
        if len(self) == 1 and self.data[0].size == 1:
            return bool(self.data[0])

        raise ValueError(
                "The truth value of an array with more than one element is "
                "ambiguous. Use actx.np.any(x) or actx.np.all(x)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __repr__(self):
        return f"DOFArray({self.data!r})"

    @classmethod
    def _serialize_init_arrays_code(cls, instance_name):
        return {"_":
                (f"{instance_name}_i", f"{instance_name}")}

    @classmethod
    def _deserialize_init_arrays_code(cls, template_instance_name, args):
        (_, arg), = args.items()
        # Why tuple([...])? https://stackoverflow.com/a/48592299
        return (f"{template_instance_name}.array_context, tuple([{arg}])")

    @property
    def size(self):
        return sum(ary.size for ary in self.data)

    @property
    def real(self):
        return DOFArray(self.array_context, tuple(subary.real for subary in self))

    @property
    def imag(self):
        return DOFArray(self.array_context, tuple(subary.imag for subary in self))


@serialize_container.register(DOFArray)
def _serialize_dof_container(ary: DOFArray):
    return list(enumerate(ary.data))


@deserialize_container.register(DOFArray)
# https://github.com/python/mypy/issues/13040
def _deserialize_dof_container(  # type: ignore[misc]
        template, iterable):
    def _raise_index_inconsistency(i, stream_i):
        raise ValueError(
                "out-of-sequence indices supplied in DOFArray deserialization "
                f"(expected {i}, received {stream_i})")

    return type(template)(
            template.array_context,
            data=tuple(
                v if i == stream_i else _raise_index_inconsistency(i, stream_i)
                for i, (stream_i, v) in enumerate(iterable)))


@with_array_context.register(DOFArray)
# https://github.com/python/mypy/issues/13040
def _with_actx_dofarray(ary: DOFArray, actx: ArrayContext) -> DOFArray:  # type: ignore[misc]
    return type(ary)(actx, ary.data)

# }}}


# {{{ nested containers

@with_container_arithmetic(bcasts_across_obj_array=False,
        eq_comparison=False, rel_comparison=False,
        _cls_has_array_context_attr=True,
        _bcast_actx_array_type=False)
@dataclass_array_container
@dataclass(frozen=True)
class MyContainer:
    name: str
    mass: DOFArray | np.ndarray
    momentum: np.ndarray
    enthalpy: DOFArray | np.ndarray

    __array_ufunc__ = None

    @property
    def array_context(self):
        if isinstance(self.mass, np.ndarray):
            return next(iter(self.mass)).array_context
        else:
            return self.mass.array_context


@with_container_arithmetic(
        bcasts_across_obj_array=False,
        container_types_bcast_across=(DOFArray, np.ndarray),
        matmul=True,
        rel_comparison=True,
        _cls_has_array_context_attr=True,
        _bcast_actx_array_type=False)
@dataclass_array_container
@dataclass(frozen=True)
class MyContainerDOFBcast:
    name: str
    mass: DOFArray | np.ndarray
    momentum: np.ndarray
    enthalpy: DOFArray | np.ndarray

    __array_ufunc__ = None

    @property
    def array_context(self):
        if isinstance(self.mass, np.ndarray):
            return next(iter(self.mass)).array_context
        else:
            return self.mass.array_context

# }}}


@with_container_arithmetic(
    bcasts_across_obj_array=True,
    rel_comparison=True,
    _cls_has_array_context_attr=True,
    _bcast_actx_array_type=False)
@dataclass_array_container
@dataclass(frozen=True)
class Foo:
    u: DOFArray

    # prevent numpy arithmetic from taking precedence
    __array_ufunc__ = None

    @property
    def array_context(self):
        return self.u.array_context


@with_container_arithmetic(bcasts_across_obj_array=True, rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class Velocity2D:
    u: ArrayContainer
    v: ArrayContainer
    array_context: ArrayContext

    __array_ufunc__ = None


@with_array_context.register(Velocity2D)
# https://github.com/python/mypy/issues/13040
def _with_actx_velocity_2d(ary, actx):  # type: ignore[misc]
    return type(ary)(ary.u, ary.v, actx)
