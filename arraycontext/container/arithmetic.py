# mypy: disallow-untyped-defs
from __future__ import annotations


__doc__ = """
.. currentmodule:: arraycontext

.. autofunction:: with_container_arithmetic
.. autoclass:: Bcast
.. autoclass:: BcastNLevels
.. autoclass:: BcastUntilActxArray

.. function:: Bcast1

    Like :class:`BcastNLevels` with *nlevels* set to 1.

.. function:: Bcast2

    Like :class:`BcastNLevels` with *nlevels* set to 2.

.. function:: Bcast3

    Like :class:`BcastNLevels` with *nlevels* set to 3.
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

import enum
from abc import ABC, abstractmethod
from dataclasses import FrozenInstanceError
from functools import partial
from numbers import Number
from typing import Any, Callable, ClassVar, Optional, Tuple, TypeVar, Union
from warnings import warn

import numpy as np

from arraycontext.context import ArrayContext, ArrayOrContainer


# {{{ with_container_arithmetic

T = TypeVar("T")


@enum.unique
class _OpClass(enum.Enum):
    ARITHMETIC = enum.auto()
    MATMUL = enum.auto()
    BITWISE = enum.auto()
    SHIFT = enum.auto()
    EQ_COMPARISON = enum.auto()
    REL_COMPARISON = enum.auto()


_UNARY_OP_AND_DUNDER = [
        ("pos", "+{}", _OpClass.ARITHMETIC),
        ("neg", "-{}", _OpClass.ARITHMETIC),
        ("abs", "abs({})", _OpClass.ARITHMETIC),
        ("inv", "~{}", _OpClass.BITWISE),
        ]
_BINARY_OP_AND_DUNDER = [
        ("add", "{} + {}", True, _OpClass.ARITHMETIC),
        ("sub", "{} - {}", True, _OpClass.ARITHMETIC),
        ("mul", "{} * {}", True, _OpClass.ARITHMETIC),
        ("truediv", "{} / {}", True, _OpClass.ARITHMETIC),
        ("floordiv", "{} // {}", True, _OpClass.ARITHMETIC),
        ("pow", "{} ** {}", True, _OpClass.ARITHMETIC),
        ("mod", "{} % {}", True, _OpClass.ARITHMETIC),
        ("divmod", "divmod({}, {})", True, _OpClass.ARITHMETIC),

        ("matmul", "{} @ {}", True, _OpClass.MATMUL),

        ("and", "{} & {}", True, _OpClass.BITWISE),
        ("or", "{} | {}", True, _OpClass.BITWISE),
        ("xor", "{} ^ {}", True, _OpClass.BITWISE),

        ("lshift", "{} << {}", False, _OpClass.SHIFT),
        ("rshift", "{} >> {}", False, _OpClass.SHIFT),

        ("eq", "{} == {}", False, _OpClass.EQ_COMPARISON),
        ("ne", "{} != {}", False, _OpClass.EQ_COMPARISON),

        ("lt", "{} < {}", False, _OpClass.REL_COMPARISON),
        ("gt", "{} > {}", False, _OpClass.REL_COMPARISON),
        ("le", "{} <= {}", False, _OpClass.REL_COMPARISON),
        ("ge", "{} >= {}", False, _OpClass.REL_COMPARISON),
        ]


def _format_unary_op_str(op_str: str, arg1: Union[Tuple[str, ...], str]) -> str:
    if isinstance(arg1, tuple):
        arg1_entry, arg1_container = arg1
        return (f"{op_str.format(arg1_entry)} "
                f"for {arg1_entry} in {arg1_container}")
    else:
        return op_str.format(arg1)


def _format_binary_op_str(op_str: str,
        arg1: Union[Tuple[str, str], str],
        arg2: Union[Tuple[str, str], str]) -> str:
    if isinstance(arg1, tuple) and isinstance(arg2, tuple):
        import sys
        if sys.version_info >= (3, 10):
            strict_arg = ", strict=__debug__"
        else:
            strict_arg = ""

        arg1_entry, arg1_container = arg1
        arg2_entry, arg2_container = arg2
        return (f"{op_str.format(arg1_entry, arg2_entry)} "
                f"for {arg1_entry}, {arg2_entry} "
                f"in zip({arg1_container}, {arg2_container}{strict_arg})")

    elif isinstance(arg1, tuple):
        arg1_entry, arg1_container = arg1
        return (f"{op_str.format(arg1_entry, arg2)} "
                f"for {arg1_entry} in {arg1_container}")

    elif isinstance(arg2, tuple):
        arg2_entry, arg2_container = arg2
        return (f"{op_str.format(arg1, arg2_entry)} "
                f"for {arg2_entry} in {arg2_container}")
    else:
        return op_str.format(arg1, arg2)


class NumpyObjectArrayMetaclass(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, np.ndarray) and instance.dtype == object


class NumpyObjectArray(metaclass=NumpyObjectArrayMetaclass):
    pass


class ComplainingNumpyNonObjectArrayMetaclass(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        if isinstance(instance, np.ndarray) and instance.dtype != object:
            # Example usage site:
            # https://github.com/illinois-ceesd/mirgecom/blob/f5d0d97c41e8c8a05546b1d1a6a2979ec8ea3554/mirgecom/inviscid.py#L148-L149
            # where normal is passed in by test_lfr_flux as a 'custom-made'
            # numpy array of dtype float64.
            warn(
                 "Broadcasting container against non-object numpy array. "
                 "This was never documented to work and will now stop working in "
                 "2025. Convert the array to an object array or use "
                 "variants of arraycontext.Bcast to obtain the desired "
                 "broadcasting semantics.", DeprecationWarning, stacklevel=3)
            return True
        else:
            return False


class ComplainingNumpyNonObjectArray(metaclass=ComplainingNumpyNonObjectArrayMetaclass):
    pass


class Bcast:
    """
    A wrapper object to force arithmetic generated by :func:`with_container_arithmetic`
    to broadcast *arg* across a container (with the container as the 'outer' structure).
    Since array containers are often nested in complex ways, different subclasses
    implement different rules on how broadcasting interacts with the hierarchy,
    with :class:`BcastNLevels` and :class:`BcastUntilActxArray` representing
    the most common.
    """
    arg: ArrayOrContainer

    # Accessing this attribute is cheaper than isinstance, so use that
    # to distinguish _BcastWithNextOperand and _BcastWithoutNextOperand.
    _with_next_operand: ClassVar[bool]

    def __init__(self, arg: ArrayOrContainer) -> None:
        object.__setattr__(self, "arg", arg)

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError()

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError()


class _BcastWithNextOperand(Bcast, ABC):
    """
    A :class:`Bcast` object that gets to see who the next operand will be, in
    order to decide whether wrapping the child in :class:`Bcast` is still necessary.
    This is much more flexible, but also considerably more expensive, than
    :class:`_BcastWithoutNextOperand`.
    """

    _with_next_operand = True

    # purposefully undocumented
    @abstractmethod
    def _rewrap(self, other_operand: ArrayOrContainer) -> ArrayOrContainer:
        ...


class _BcastWithoutNextOperand(Bcast, ABC):
    """
    A :class:`Bcast` object that does not get to see who the next operand will be.
    """
    _with_next_operand = False

    # purposefully undocumented
    @abstractmethod
    def _rewrap(self) -> ArrayOrContainer:
        ...


class BcastNLevels(_BcastWithoutNextOperand):
    """
    A broadcasting rule that lets *arg* broadcast against *nlevels* "levels" of
    array containers. Use :func:`Bcast1`, :func:`Bcast2`, :func:`Bcast3` as
    convenient aliases for the common cases.

    Usage example::

        container + Bcast2(actx_array)

    .. note::

        :mod:`numpy` object arrays do not count against the number of levels.

    .. automethod:: __init__
    """
    nlevels: int

    def __init__(self, nlevels: int, arg: ArrayOrContainer) -> None:
        if nlevels < 1:
            raise ValueError("nlevels is expected to be one or greater.")

        super().__init__(arg)
        object.__setattr__(self, "nlevels", nlevels)

    def _rewrap(self) -> ArrayOrContainer:
        if self.nlevels == 1:
            return self.arg
        else:
            return BcastNLevels(self.nlevels-1, self.arg)


Bcast1 = partial(BcastNLevels, 1)
Bcast2 = partial(BcastNLevels, 2)
Bcast3 = partial(BcastNLevels, 3)


class BcastUntilActxArray(_BcastWithNextOperand):
    """
    A broadcast rule that broadcasts *arg* across array containers until
    the 'opposite' operand is one of the :attr:`~arraycontext.ArrayContext.array_types`
    of *actx*, or a :class:`~numbers.Number`.

    Suggested usage pattern::

        bcast = functools.partial(BcastUntilActxArray, actx)

        container + bcast(actx_array)

    .. automethod:: __init__
    """
    actx: ArrayContext

    def __init__(self,
                 actx: ArrayContext,
                 arg: ArrayOrContainer) -> None:
        super().__init__(arg)
        object.__setattr__(self, "actx", actx)

    def _rewrap(self, other_operand: ArrayOrContainer) -> ArrayOrContainer:
        if isinstance(other_operand, (*self.actx.array_types, Number)):
            return self.arg
        else:
            return BcastUntilActxArray(self.actx, self.arg)


def with_container_arithmetic(
        *,
        bcast_number: bool = True,
        _bcast_actx_array_type: Optional[bool] = None,
        bcast_obj_array: Optional[bool] = None,
        bcast_numpy_array: bool = False,
        bcast_container_types: Optional[Tuple[type, ...]] = None,
        arithmetic: bool = True,
        matmul: bool = False,
        bitwise: bool = False,
        shift: bool = False,
        _cls_has_array_context_attr: Optional[bool] = None,
        eq_comparison: Optional[bool] = None,
        rel_comparison: Optional[bool] = None) -> Callable[[type], type]:
    """A class decorator that implements built-in operators for array containers
    by propagating the operations to the elements of the container.

    :arg bcast_number: If *True*, numbers broadcast over the container
        (with the container as the 'outer' structure).
    :arg bcast_obj_array: If *True*, this container will be broadcast
        across :mod:`numpy` object arrays
        (with the object array as the 'outer' structure).
        Add :class:`numpy.ndarray` to *bcast_container_types* to achieve
        the 'reverse' broadcasting.
    :arg bcast_container_types: A sequence of container types that will broadcast
        across this container, with this container as the 'outer' structure.
        :class:`numpy.ndarray` is permitted to be part of this sequence to
        indicate that object arrays (and *only* object arrays) will be broadcasat.
        In this case, *bcast_obj_array* must be *False*.
    :arg arithmetic: Implement the conventional arithmetic operators, including
        ``**``, :func:`divmod`, and ``//``. Also includes ``+`` and ``-`` as well as
        :func:`abs`.
    :arg bitwise: If *True*, implement bitwise and, or, not, and inversion.
    :arg shift: If *True*, implement bit shifts.
    :arg eq_comparison: If *True*, implement ``==`` and ``!=``.
    :arg rel_comparison: If *True*, implement ``<``, ``<=``, ``>``, ``>=``.
        In that case, if *eq_comparison* is unspecified, it is also set to
        *True*.
    :arg _cls_has_array_context_attr: A flag indicating whether the decorated
        class has an ``array_context`` attribute. If so, and if :data:`__debug__`
        is *True*, an additional check is performed in binary operators
        to ensure that both containers use the same array context.
        If *None* (the default), this value is set based on whether the class
        has an ``array_context`` attribute.
        Consider this argument an unstable interface. It may disappear at any moment.

    Each operator class also includes the "reverse" operators if applicable.

    .. note::

        For the generated binary arithmetic operators, if certain types
        should be broadcast over the container (with the container as the
        'outer' structure) but are not handled in this way by their types,
        you may wrap them in one of the :class:`Bcast` variants to achieve
        the desired semantics.

    .. note::

        To generate the code implementing the operators, this function relies on
        class methods ``_deserialize_init_arrays_code`` and
        ``_serialize_init_arrays_code``. This interface should be considered
        undocumented and subject to change, however if you are curious, you may look
        at its implementation in :class:`meshmode.dof_array.DOFArray`. For a simple
        structure type, the implementation might look like this::

            @classmethod
            def _serialize_init_arrays_code(cls, instance_name):
                return {"u": f"{instance_name}.u", "v": f"{instance_name}.v"}

            @classmethod
            def _deserialize_init_arrays_code(cls, tmpl_instance_name, args):
                return f"u={args['u']}, v={args['v']}"

    :func:`dataclass_array_container` automatically generates an appropriate
    implementation of these methods, so :func:`with_container_arithmetic`
    should nest "outside" :func:dataclass_array_container`.
    """

    # Hard-won design lessons:
    #
    # - Anything that special-cases np.ndarray by type is broken by design because:
    #   - np.ndarray is an array context array.
    #   - numpy object arrays can be array containers.
    #   Using NumpyObjectArray and NumpyNonObjectArray *may* be better?
    #   They're new, so there is no operational experience with them.
    #
    # - Broadcast rules are hard to change once established, particularly
    #   because one cannot grep for their use.
    #
    # Possible advantages of the "Bcast" broadcast-rule-as-object design:
    #
    # - If one rule does not fit the user's need, they can straightforwardly use
    #   another.
    #
    # - It's straightforward to find where certain broadcast rules are used.
    #
    # - The broadcast rule can contain more state. For example, it's now easy
    #   for the rule to know what array context should be used to determine
    #   actx array types.
    #
    # Possible downsides of the "Bcast" broadcast-rule-as-object design:
    #
    # - User code is a bit more wordy.
    #
    # - Rewrapping has the potential to be costly, especially in
    #   _with_next_operand mode.

    # {{{ handle inputs

    if bcast_obj_array is None:
        raise TypeError("bcast_obj_array must be specified")

    if rel_comparison is None:
        raise TypeError("rel_comparison must be specified")

    if bcast_numpy_array:
        warn("'bcast_numpy_array=True' is deprecated and will be unsupported"
             " from 2025.", DeprecationWarning, stacklevel=2)

        if _bcast_actx_array_type:
            raise ValueError("'bcast_numpy_array' and '_bcast_actx_array_type'"
                             " cannot be both set.")

    if rel_comparison and eq_comparison is None:
        eq_comparison = True

    if eq_comparison is None:
        raise TypeError("eq_comparison must be specified")

    if not bcast_obj_array and bcast_numpy_array:
        raise TypeError("bcast_obj_array must be set if bcast_numpy_array is")

    if bcast_numpy_array:
        def numpy_pred(name: str) -> str:
            return f"is_numpy_array({name})"
    elif bcast_obj_array:
        def numpy_pred(name: str) -> str:
            return f"isinstance({name}, np.ndarray) and {name}.dtype.char == 'O'"
    else:
        def numpy_pred(name: str) -> str:
            return "False"  # optimized away

    if bcast_container_types is None:
        bcast_container_types = ()

    if np.ndarray in bcast_container_types and bcast_obj_array:
        raise ValueError("If numpy.ndarray is part of bcast_container_types, "
                "bcast_obj_array must be False.")

    numpy_check_types: list[type] = [NumpyObjectArray, ComplainingNumpyNonObjectArray]
    bcast_container_types = tuple(
        new_ct
        for old_ct in bcast_container_types
        for new_ct in
        (numpy_check_types
        if old_ct is np.ndarray
        else [old_ct])
    )

    desired_op_classes = set()
    if arithmetic:
        desired_op_classes.add(_OpClass.ARITHMETIC)
    if matmul:
        desired_op_classes.add(_OpClass.MATMUL)
    if bitwise:
        desired_op_classes.add(_OpClass.BITWISE)
    if shift:
        desired_op_classes.add(_OpClass.SHIFT)
    if eq_comparison:
        desired_op_classes.add(_OpClass.EQ_COMPARISON)
    if rel_comparison:
        desired_op_classes.add(_OpClass.REL_COMPARISON)

    # }}}

    def wrap(cls: Any) -> Any:
        if not hasattr(cls, "__array_ufunc__"):
            warn(f"{cls} does not have __array_ufunc__ set. "
                 "This will cause numpy to attempt broadcasting, in a way that "
                 "is likely undesired. "
                 f"To avoid this, set __array_ufunc__ = None in {cls}.",
                 stacklevel=2)

        cls_has_array_context_attr: bool | None = _cls_has_array_context_attr
        bcast_actx_array_type: bool | None = _bcast_actx_array_type

        if cls_has_array_context_attr is None:
            if hasattr(cls, "array_context"):
                raise TypeError(
                        f"{cls} has an 'array_context' attribute, but it does not "
                        "set '_cls_has_array_context_attr' to *True* when calling "
                        "with_container_arithmetic. This is being interpreted "
                        "as '.array_context' being permitted to fail "
                        "with an exception, which is no longer allowed. "
                        f"If {cls.__name__}.array_context will not fail, pass "
                        "'_cls_has_array_context_attr=True'. "
                        "If you do not want container arithmetic to make "
                        "use of the array context, set "
                        "'_cls_has_array_context_attr=False'.")

        if bcast_actx_array_type is None:
            if cls_has_array_context_attr:
                if bcast_number:
                    bcast_actx_array_type = cls_has_array_context_attr
            else:
                bcast_actx_array_type = False
        else:
            if bcast_actx_array_type and not cls_has_array_context_attr:
                raise TypeError("_bcast_actx_array_type can be True only if "
                                "_cls_has_array_context_attr is set.")

        if bcast_actx_array_type:
            if _bcast_actx_array_type:
                warn(
                    f"Broadcasting array context array types across {cls} "
                    "has been explicitly "
                    "enabled. As of 2025, this will stop working. "
                    "Express these operations using arraycontext.Bcast variants "
                    "instead."
                    "To opt out now (and avoid this warning), "
                    "pass _bcast_actx_array_type=False. ",
                    DeprecationWarning, stacklevel=2)
            else:
                warn(
                    f"Broadcasting array context array types across {cls} "
                    "has been implicitly "
                    "enabled. As of 2025, this will no longer work. "
                    "Express these operations using arraycontext.Bcast variants "
                    "instead."
                    "To opt out now (and avoid this warning), "
                    "pass _bcast_actx_array_type=False.",
                    DeprecationWarning, stacklevel=2)

        if (not hasattr(cls, "_serialize_init_arrays_code")
                or not hasattr(cls, "_deserialize_init_arrays_code")):
            raise TypeError(f"class '{cls.__name__}' must provide serialization "
                    "code to generate arithmetic operations by implementing "
                    "'_serialize_init_arrays_code' and "
                    "'_deserialize_init_arrays_code'. If this is a dataclass, "
                    "use the 'dataclass_array_container' decorator first.")

        from pytools.codegen import CodeGenerator, Indentation
        gen = CodeGenerator()
        gen(f"""
            from numbers import Number
            import numpy as np
            from arraycontext import ArrayContainer, Bcast
            from warnings import warn

            def _raise_if_actx_none(actx):
                if actx is None:
                    raise ValueError("array containers with frozen arrays "
                        "cannot be operated upon")
                return actx

            def is_numpy_array(arg):
                if isinstance(arg, np.ndarray):
                    if arg.dtype != "O":
                        warn("Operand is a non-object numpy array, "
                            "and the broadcasting behavior of this array container "
                            "({cls}) "
                            "is influenced by this because of its use of "
                            "the deprecated bcast_numpy_array. This broadcasting "
                            "behavior will change in 2025. If you would like the "
                            "broadcasting behavior to stay the same, make sure "
                            "to convert the passed numpy array to an "
                            "object array, or use arraycontext.Bcast to achieve "
                            "the desired broadcasting semantics.",
                            DeprecationWarning, stacklevel=2)
                    return True
                else:
                    return False

            """)
        gen("")

        if bcast_container_types:
            for i, bct in enumerate(bcast_container_types):
                gen(f"from {bct.__module__} import {bct.__qualname__} as _bctype{i}")
            gen("")
        outer_bcast_type_names = tuple([
                f"_bctype{i}" for i in range(len(bcast_container_types))
                ])
        if bcast_number:
            outer_bcast_type_names += ("Number",)

        def same_key(k1: T, k2: T) -> T:
            assert k1 == k2
            return k1

        def tup_str(t: Tuple[str, ...]) -> str:
            if not t:
                return "()"
            else:
                return "({},)".format(", ".join(t))

        gen(f"cls._outer_bcast_types = {tup_str(outer_bcast_type_names)}")
        gen(f"cls._bcast_numpy_array = {bcast_numpy_array}")
        gen(f"cls._bcast_obj_array = {bcast_obj_array}")
        gen("")

        # {{{ unary operators

        for dunder_name, op_str, op_cls in _UNARY_OP_AND_DUNDER:
            if op_cls not in desired_op_classes:
                continue

            fname = f"_{cls.__name__.lower()}_{dunder_name}"
            init_args = cls._deserialize_init_arrays_code("arg1", {
                    key_arg1: _format_unary_op_str(op_str, expr_arg1)
                    for key_arg1, expr_arg1 in
                    cls._serialize_init_arrays_code("arg1").items()
                    })

            gen(f"""
                def {fname}(arg1):
                    return cls({init_args})
                cls.__{dunder_name}__ = {fname}""")
            gen("")

        # }}}

        # {{{ binary operators

        for dunder_name, op_str, reversible, op_cls in _BINARY_OP_AND_DUNDER:
            fname = f"_{cls.__name__.lower()}_{dunder_name}"

            if op_cls not in desired_op_classes:
                # Leaving equality comparison at the default supplied by
                # dataclasses is dangerous: Comparison of dataclass fields
                # might return an array of truth values, and the dataclasses
                # implementation of __eq__ might consider that 'truthy' enough,
                # yielding bogus equality results.
                if op_cls == _OpClass.EQ_COMPARISON:
                    gen(f"def {fname}(arg1, arg2):")
                    with Indentation(gen):
                        gen("return NotImplemented")
                    gen(f"cls.__{dunder_name}__ = {fname}")
                    gen("")

                continue

            zip_init_args = cls._deserialize_init_arrays_code("arg1", {
                    same_key(key_arg1, key_arg2):
                    _format_binary_op_str(op_str, expr_arg1, expr_arg2)
                    for (key_arg1, expr_arg1), (key_arg2, expr_arg2) in zip(
                        cls._serialize_init_arrays_code("arg1").items(),
                        cls._serialize_init_arrays_code("arg2").items())
                    })
            bcast_init_args_arg1_is_outer = cls._deserialize_init_arrays_code("arg1", {
                    key_arg1: _format_binary_op_str(op_str, expr_arg1, "arg2")
                    for key_arg1, expr_arg1 in
                    cls._serialize_init_arrays_code("arg1").items()
                    })
            bcast_init_args_arg2_is_outer = cls._deserialize_init_arrays_code("arg2", {
                    key_arg2: _format_binary_op_str(op_str, "arg1", expr_arg2)
                    for key_arg2, expr_arg2 in
                    cls._serialize_init_arrays_code("arg2").items()
                    })

            def get_operand(arg: Union[tuple[str, str], str]) -> str:
                if isinstance(arg, tuple):
                    entry, _container = arg
                    return entry
                else:
                    return arg

            bcast_init_args_arg1_is_outer_with_rewrap = \
                cls._deserialize_init_arrays_code("arg1", {
                    key_arg1:
                    _format_binary_op_str(
                            op_str, expr_arg1,
                            f"arg2._rewrap({get_operand(expr_arg1)})")
                    for key_arg1, expr_arg1 in
                    cls._serialize_init_arrays_code("arg1").items()
                    })
            bcast_init_args_arg2_is_outer_with_rewrap = \
                cls._deserialize_init_arrays_code("arg2", {
                    key_arg2:
                    _format_binary_op_str(
                            op_str,
                            f"arg1._rewrap({get_operand(expr_arg2)})",
                            expr_arg2)
                    for key_arg2, expr_arg2 in
                    cls._serialize_init_arrays_code("arg2").items()
                    })

            # {{{ "forward" binary operators

            gen(f"def {fname}(arg1, arg2):")
            with Indentation(gen):
                gen("if arg2.__class__ is cls:")
                with Indentation(gen):
                    if __debug__ and cls_has_array_context_attr:
                        gen("""
                            arg1_actx = arg1.array_context
                            arg2_actx = arg2.array_context
                            if arg1_actx is not arg2_actx:
                                msg = ("array contexts of both arguments "
                                    "must match")
                                if arg1_actx is None:
                                    raise ValueError(msg
                                        + ": left operand is frozen "
                                        "(i.e. has no array context)")
                                elif arg2_actx is None:
                                    raise ValueError(msg
                                        + ": right operand is frozen "
                                        "(i.e. has no array context)")
                                else:
                                    raise ValueError(msg)""")
                    gen(f"return cls({zip_init_args})")

                if bcast_actx_array_type:
                    if __debug__:
                        bcast_actx_ary_types: tuple[str, ...] = (
                            "*_raise_if_actx_none("
                            "arg1.array_context).array_types",)
                    else:
                        bcast_actx_ary_types = (
                                "*arg1.array_context.array_types",)
                else:
                    bcast_actx_ary_types = ()

                gen(f"""
                if {numpy_pred("arg2")}:
                    result = np.empty_like(arg2, dtype=object)
                    for i in np.ndindex(arg2.shape):
                        result[i] = {op_str.format("arg1", "arg2[i]")}
                    return result

                if {bool(outer_bcast_type_names)}:  # optimized away
                    if isinstance(arg2,
                                  {tup_str(outer_bcast_type_names
                                           + bcast_actx_ary_types)}):
                        if __debug__:
                            if isinstance(arg2, {tup_str(bcast_actx_ary_types)}):
                                warn("Broadcasting {cls} over array "
                                    f"context array type {{type(arg2)}} is deprecated "
                                    "and will no longer work in 2025. "
                                    "Use arraycontext.Bcast to achieve the desired "
                                    "broadcasting semantics.",
                                    DeprecationWarning, stacklevel=2)

                        return cls({bcast_init_args_arg1_is_outer})

                if isinstance(arg2, Bcast):
                    if arg2._with_next_operand:
                        return cls({bcast_init_args_arg1_is_outer_with_rewrap})
                    else:
                        arg2 = arg2._rewrap()
                        return cls({bcast_init_args_arg1_is_outer})

                return NotImplemented
                """)
            gen(f"cls.__{dunder_name}__ = {fname}")
            gen("")

            # }}}

            # {{{ "reverse" binary operators

            if reversible:
                fname = f"_{cls.__name__.lower()}_r{dunder_name}"

                if bcast_actx_array_type:
                    if __debug__:
                        bcast_actx_ary_types = (
                            "*_raise_if_actx_none("
                            "arg2.array_context).array_types",)
                    else:
                        bcast_actx_ary_types = (
                                "*arg2.array_context.array_types",)
                else:
                    bcast_actx_ary_types = ()

                gen(f"""
                    def {fname}(arg2, arg1):
                        # assert other.__cls__ is not cls

                        if {numpy_pred("arg1")}:
                            result = np.empty_like(arg1, dtype=object)
                            for i in np.ndindex(arg1.shape):
                                result[i] = {op_str.format("arg1[i]", "arg2")}
                            return result
                        if {bool(outer_bcast_type_names)}:  # optimized away
                            if isinstance(arg1,
                                          {tup_str(outer_bcast_type_names
                                                   + bcast_actx_ary_types)}):
                                if __debug__:
                                    if isinstance(arg1,
                                            {tup_str(bcast_actx_ary_types)}):
                                        warn("Broadcasting {cls} over array "
                                            f"context array type {{type(arg1)}} "
                                            "is deprecated "
                                            "and will no longer work in 2025.",
                                            "Use arraycontext.Bcast to achieve the "
                                            "desired broadcasting semantics.",
                                            DeprecationWarning, stacklevel=2)

                                return cls({bcast_init_args_arg2_is_outer})

                        if isinstance(arg1, Bcast):
                            if arg1._with_next_operand:
                                return cls({bcast_init_args_arg2_is_outer_with_rewrap})
                            else:
                                arg1 = arg1._rewrap()
                                return cls({bcast_init_args_arg2_is_outer})

                        return NotImplemented

                    cls.__r{dunder_name}__ = {fname}""")
                gen("")

            # }}}

        # }}}

        # This will evaluate the module, which is all we need.
        code = gen.get().rstrip()+"\n"

        result_dict = {"_MODULE_SOURCE_CODE": code, "cls": cls}
        exec(compile(code, f"<container arithmetic for {cls.__name__}>", "exec"),
                result_dict)

        return cls

    # we're being called as @with_container_arithmetic(...), with parens
    return wrap

# }}}


# vim: foldmethod=marker
