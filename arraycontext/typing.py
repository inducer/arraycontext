"""
.. currentmodule:: arraycontext

Types and Type Variables for Arrays and Containers
--------------------------------------------------

.. autoclass:: ScalarLike

    A type alias of :data:`pymbolic.Scalar`.

.. autoclass:: Array

.. autoclass:: ArrayT

    A type variable with a lower bound of :class:`Array`.

See also :class:`ArrayContainer` and :class:`ArrayOrContainerT`.

.. autoclass:: ArrayOrScalar
.. autoclass:: ArrayOrScalarT
.. autoclass:: ArrayOrContainer
.. autoclass:: ArrayOrContainerT

    A type variable with a bound of :class:`ArrayOrContainer`.

.. autoclass:: ArrayOrArithContainer
.. autoclass:: ArrayOrArithContainerT
.. autoclass:: ContainerOrScalarT
.. autoclass:: ArrayOrArithContainerOrScalar
.. autoclass:: ArrayOrArithContainerOrScalarT

    A type variable with a bound of :class:`ArrayOrContainerOrScalar`.

.. autoclass:: ArrayOrContainerOrScalar

.. autoclass:: ArrayOrContainerOrScalarT

    A type variable with a bound of :class:`ArrayOrContainerOrScalar`.

Other locations
---------------
.. currentmodule:: arraycontext.typing

.. class:: ArrayContainerT

    :canonical: :class:`arraycontext.ArrayContainerT`.
"""
from __future__ import annotations


__copyright__ = """
Copyright (C) 2025 University of Illinois Board of Trustees
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
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    SupportsInt,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from typing_extensions import Self, TypeIs

from pymbolic.typing import Integer, Scalar as _Scalar
from pytools.obj_array import ObjectArrayND


if TYPE_CHECKING:
    from numpy.typing import DTypeLike


# deprecated, use ScalarLike instead
Scalar: TypeAlias = _Scalar
ScalarLike = Scalar
ScalarLikeT = TypeVar("ScalarLikeT", bound=ScalarLike)

# {{{ array

# We won't support 'A' and 'K', since they depend on in-memory order; that is
# not intended to be a meaningful concept for actx arrays.
OrderCF: TypeAlias = Literal["C"] | Literal["F"]


class Array(Protocol):
    """A :class:`~typing.Protocol` for the array type supported by
    :class:`ArrayContext`.

    This is meant to aid in typing annotations. For a explicit list of
    supported types see :attr:`ArrayContext.array_types`.

    .. attribute:: shape
    .. attribute:: size
    .. attribute:: dtype
    .. attribute:: __getitem__

    In addition, arrays are expected to support basic arithmetic.
    """

    @property
    def shape(self) -> tuple[Array | Integer, ...]:
        ...

    @property
    def size(self) -> Array | Integer:
        ...

    def __len__(self) -> int: ...

    @property
    def dtype(self) -> np.dtype[Any]:
        ...

    # Covering all the possible index variations is hard and (kind of) futile.
    # If you'd  like to see how, try changing the Any to
    # AxisIndex = slice | int | "Array"
    # Index = AxisIndex |tuple[AxisIndex]
    def __getitem__(self, index: Any) -> Array:  # pyright: ignore[reportAny]
        ...

    # Some basic arithmetic that's supposed to work
    # Need to return Array instead of Self because for some array types, arithmetic
    # operations on one subtype may result in a different subtype.
    # For example, pytato arrays: <Placeholder> + 1 -> <IndexLambda>
    def __neg__(self) -> Array: ...
    def __abs__(self) -> Array: ...
    def __add__(self, other: Self | ScalarLike) -> Array: ...
    def __radd__(self, other: Self | ScalarLike) -> Array: ...
    def __sub__(self, other: Self | ScalarLike) -> Array: ...
    def __rsub__(self, other: Self | ScalarLike) -> Array: ...
    def __mul__(self, other: Self | ScalarLike) -> Array: ...
    def __rmul__(self, other: Self | ScalarLike) -> Array: ...
    def __pow__(self, other: Self | ScalarLike) -> Array: ...
    def __rpow__(self, other: Self | ScalarLike) -> Array: ...
    def __truediv__(self, other: Self | ScalarLike) -> Array: ...
    def __rtruediv__(self, other: Self | ScalarLike) -> Array: ...

    def copy(self) -> Self: ...

    @property
    def real(self) -> Array: ...
    @property
    def imag(self) -> Array: ...
    def conj(self) -> Array: ...

    def astype(self, dtype: DTypeLike) -> Array: ...

    # Annoyingly, numpy 2.3.1 (and likely earlier) treats these differently when
    # reshaping to the empty shape (), so we need to expose both.
    @overload
    def reshape(self, *shape: int, order: OrderCF = "C") -> Array: ...

    @overload
    def reshape(self, shape: tuple[int, ...], /, *, order: OrderCF = "C") -> Array: ...

    @property
    def T(self) -> Array: ...  # noqa: N802

    def transpose(self, axes: tuple[int, ...]) -> Array: ...

# }}}


# {{{ array container

class _UserDefinedArrayContainer(Protocol):
    # This is used as a type annotation in dataclasses that are processed
    # by dataclass_array_container, where it's used to recognize attributes
    # that are container-typed.

    # This method prevents ArrayContainer from matching any object, while
    # matching numpy object arrays and many array containers.
    __array_ufunc__: ClassVar[None]


ArrayContainer: TypeAlias = (
    ObjectArrayND["ArrayOrContainerOrScalar"]
    | _UserDefinedArrayContainer
    )


class _UserDefinedArithArrayContainer(_UserDefinedArrayContainer, Protocol):
    # This is loose and permissive, assuming that any array can be added
    # to any container. The alternative would be to plaster type-ignores
    # on all those uses. Achieving typing precision on what broadcasting is
    # allowable seems like a huge endeavor and is likely not feasible without
    # a mypy plugin. Maybe some day? -AK, November 2024

    def __neg__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def __add__(self, other: ArrayOrScalar | Self) -> Self: ...
    def __radd__(self, other: ArrayOrScalar | Self) -> Self: ...
    def __sub__(self, other: ArrayOrScalar | Self) -> Self: ...
    def __rsub__(self, other: ArrayOrScalar | Self) -> Self: ...
    def __mul__(self, other: ArrayOrScalar | Self) -> Self: ...
    def __rmul__(self, other: ArrayOrScalar | Self) -> Self: ...
    def __truediv__(self, other: ArrayOrScalar | Self) -> Self: ...
    def __rtruediv__(self, other: ArrayOrScalar | Self) -> Self: ...
    def __pow__(self, other: ArrayOrScalar | Self) -> Self: ...
    def __rpow__(self, other: ArrayOrScalar | Self) -> Self: ...


ArithArrayContainer: TypeAlias = (
    ObjectArrayND["ArrayOrArithContainerOrScalar"]
    | _UserDefinedArithArrayContainer)


ArrayContainerT = TypeVar("ArrayContainerT", bound=ArrayContainer)

# }}}


ArrayT = TypeVar("ArrayT", bound=Array)
ArrayOrScalar: TypeAlias = Array | _Scalar
ArrayOrScalarT = TypeVar("ArrayOrScalarT", bound=ArrayOrScalar)
ArrayOrContainer: TypeAlias = Array | ArrayContainer
ArrayOrArithContainer: TypeAlias = Array | ArithArrayContainer
ArrayOrArithContainerTc = TypeVar("ArrayOrArithContainerTc",
                                 Array, "ArithArrayContainer")
ArrayOrContainerT = TypeVar("ArrayOrContainerT", bound=ArrayOrContainer)
ArrayOrArithContainerT = TypeVar("ArrayOrArithContainerT", bound=ArrayOrArithContainer)
ArrayOrContainerOrScalar: TypeAlias = Array | ArrayContainer | ScalarLike
ArrayOrArithContainerOrScalar: TypeAlias = Array | ArithArrayContainer | ScalarLike
ArrayOrContainerOrScalarT = TypeVar(
        "ArrayOrContainerOrScalarT",
        bound=ArrayOrContainerOrScalar)
ArrayOrArithContainerOrScalarT = TypeVar(
        "ArrayOrArithContainerOrScalarT",
        bound=ArrayOrArithContainerOrScalar)


ContainerOrScalarT = TypeVar("ContainerOrScalarT", bound="ArrayContainer | ScalarLike")


NumpyOrContainerOrScalar: TypeAlias = "np.ndarray | ArrayContainer | ScalarLike"


def is_scalar_like(x: object, /) -> TypeIs[Scalar]:
    return np.isscalar(x)


def shape_is_int_only(shape: tuple[Array | Integer, ...], /) -> tuple[int, ...]:
    res: list[int] = []
    for i, s in enumerate(shape):
        try:
            res.append(int(cast("SupportsInt", s)))
        except TypeError:
            raise TypeError(
                    "only non-parametric shapes are allowed in this context, "
                    f"axis {i+1} is {type(s)}"
                ) from None

    return tuple(res)
