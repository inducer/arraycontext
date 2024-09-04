# mypy: disallow-untyped-defs
from __future__ import annotations


__doc__ = """
.. currentmodule:: arraycontext

.. autofunction:: with_container_arithmetic
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
from typing import Any, Callable, Optional, Tuple, TypeVar, Union
from warnings import warn

import numpy as np


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
                 "2025. Convert the array to an object array to preserve the "
                 "current semantics.", DeprecationWarning, stacklevel=3)
            return True
        else:
            return False


class ComplainingNumpyNonObjectArray(metaclass=ComplainingNumpyNonObjectArrayMetaclass):
    pass


def with_container_arithmetic(
            *,
            number_bcasts_across: Optional[bool] = None,
            bcasts_across_obj_array: Optional[bool] = None,
            container_types_bcast_across: Optional[Tuple[type, ...]] = None,
            arithmetic: bool = True,
            matmul: bool = False,
            bitwise: bool = False,
            shift: bool = False,
            _cls_has_array_context_attr: Optional[bool] = None,
            eq_comparison: Optional[bool] = None,
            rel_comparison: Optional[bool] = None,

            # deprecated:
            bcast_number: Optional[bool] = None,
            bcast_obj_array: Optional[bool] = None,
            bcast_numpy_array: bool = False,
            _bcast_actx_array_type: Optional[bool] = None,
            bcast_container_types: Optional[Tuple[type, ...]] = None,
        ) -> Callable[[type], type]:
    """A class decorator that implements built-in operators for array containers
    by propagating the operations to the elements of the container.

    :arg number_bcasts_across: If *True*, numbers broadcast over the container
        (with the container as the 'outer' structure).
    :arg bcasts_across_obj_array: If *True*, this container will be broadcast
        across :mod:`numpy` object arrays
        (with the object array as the 'outer' structure).
        Add :class:`numpy.ndarray` to *container_types_bcast_across* to achieve
        the 'reverse' broadcasting.
    :arg container_types_bcast_across: A sequence of container types that will broadcast
        across this container, with this container as the 'outer' structure.
        :class:`numpy.ndarray` is permitted to be part of this sequence to
        indicate that object arrays (and *only* object arrays) will be broadcast.
        In this case, *bcasts_across_obj_array* must be *False*.
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

    # {{{ handle inputs

    if rel_comparison and eq_comparison is None:
        eq_comparison = True

    if eq_comparison is None:
        raise TypeError("eq_comparison must be specified")

    # {{{ handle bcast_number

    if bcast_number is not None:
        if number_bcasts_across is not None:
            raise TypeError(
                    "may specify at most one of 'bcast_number' and "
                    "'number_bcasts_across'")

        warn("'bcast_number' is deprecated and will be unsupported from 2025. "
             "Use 'number_bcasts_across', with equivalent meaning.",
              DeprecationWarning, stacklevel=2)
        number_bcasts_across = bcast_number
    else:
        if number_bcasts_across is None:
            number_bcasts_across = True

    del bcast_number

    # }}}

    # {{{ handle bcast_obj_array

    if bcast_obj_array is not None:
        if bcasts_across_obj_array is not None:
            raise TypeError(
                    "may specify at most one of 'bcast_obj_array' and "
                    "'bcasts_across_obj_array'")

        warn("'bcast_obj_array' is deprecated and will be unsupported from 2025. "
             "Use 'bcasts_across_obj_array', with equivalent meaning.",
              DeprecationWarning, stacklevel=2)
        bcasts_across_obj_array = bcast_obj_array
    else:
        if bcasts_across_obj_array is None:
            raise TypeError("bcasts_across_obj_array must be specified")

    del bcast_obj_array

    # }}}

    # {{{ handle bcast_container_types

    if bcast_container_types is not None:
        if container_types_bcast_across is not None:
            raise TypeError(
                    "may specify at most one of 'bcast_container_types' and "
                    "'container_types_bcast_across'")

        warn("'bcast_container_types' is deprecated and will be unsupported from 2025. "
             "Use 'container_types_bcast_across', with equivalent meaning.",
              DeprecationWarning, stacklevel=2)
        container_types_bcast_across = bcast_container_types
    else:
        if container_types_bcast_across is None:
            container_types_bcast_across = ()

    del bcast_container_types

    # }}}

    if rel_comparison is None:
        raise TypeError("rel_comparison must be specified")

    if bcast_numpy_array:
        warn("'bcast_numpy_array=True' is deprecated and will be unsupported"
             " from 2025.", DeprecationWarning, stacklevel=2)

        if _bcast_actx_array_type:
            raise ValueError("'bcast_numpy_array' and '_bcast_actx_array_type'"
                             " cannot be both set.")

    if not bcasts_across_obj_array and bcast_numpy_array:
        raise TypeError("bcast_obj_array must be set if bcast_numpy_array is")

    if bcast_numpy_array:
        def numpy_pred(name: str) -> str:
            return f"is_numpy_array({name})"
    elif bcasts_across_obj_array:
        def numpy_pred(name: str) -> str:
            return f"isinstance({name}, np.ndarray) and {name}.dtype.char == 'O'"
    else:
        def numpy_pred(name: str) -> str:
            return "False"  # optimized away

    if np.ndarray in container_types_bcast_across and bcasts_across_obj_array:
        raise ValueError("If numpy.ndarray is part of bcast_container_types, "
                "bcast_obj_array must be False.")

    numpy_check_types: list[type] = [NumpyObjectArray, ComplainingNumpyNonObjectArray]
    container_types_bcast_across = tuple(
        new_ct
        for old_ct in container_types_bcast_across
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
                if number_bcasts_across:
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
                    "There is no replacement as of right now. "
                    "See the discussion in "
                    "https://github.com/inducer/arraycontext/pull/190. "
                    "To opt out now (and avoid this warning), "
                    "pass _bcast_actx_array_type=False. ",
                    DeprecationWarning, stacklevel=2)
            else:
                warn(
                    f"Broadcasting array context array types across {cls} "
                    "has been implicitly "
                    "enabled. As of 2025, this will no longer work. "
                    "There is no replacement as of right now. "
                    "See the discussion in "
                    "https://github.com/inducer/arraycontext/pull/190. "
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
            from arraycontext import ArrayContainer
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
                            "object array.",
                            DeprecationWarning, stacklevel=3)
                    return True
                else:
                    return False

            """)
        gen("")

        if container_types_bcast_across:
            for i, bct in enumerate(container_types_bcast_across):
                gen(f"from {bct.__module__} import {bct.__qualname__} as _bctype{i}")
            gen("")
        container_type_names_bcast_across = tuple(
                f"_bctype{i}" for i in range(len(container_types_bcast_across)))
        if number_bcasts_across:
            container_type_names_bcast_across += ("Number",)

        def same_key(k1: T, k2: T) -> T:
            assert k1 == k2
            return k1

        def tup_str(t: Tuple[str, ...]) -> str:
            if not t:
                return "()"
            else:
                return "({},)".format(", ".join(t))

        gen(f"cls._outer_bcast_types = {tup_str(container_type_names_bcast_across)}")
        gen("cls._container_types_bcast_across = "
            f"{tup_str(container_type_names_bcast_across)}")

        gen(f"cls._bcast_numpy_array = {bcast_numpy_array}")

        gen(f"cls._bcast_obj_array = {bcasts_across_obj_array}")
        gen(f"cls._bcasts_across_obj_array = {bcasts_across_obj_array}")
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

                if {bool(container_type_names_bcast_across)}:  # optimized away
                    if isinstance(arg2,
                                  {tup_str(container_type_names_bcast_across
                                           + bcast_actx_ary_types)}):
                        if __debug__:
                            if isinstance(arg2, {tup_str(bcast_actx_ary_types)}):
                                warn("Broadcasting {cls} over array "
                                    f"context array type {{type(arg2)}} is deprecated "
                                    "and will no longer work in 2025. "
                                    "There is no replacement as of right now. "
                                    "See the discussion in "
                                    "https://github.com/inducer/arraycontext/"
                                    "pull/190. ",
                                    DeprecationWarning, stacklevel=2)

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
                        if {bool(container_type_names_bcast_across)}:  # optimized away
                            if isinstance(arg1,
                                          {tup_str(container_type_names_bcast_across
                                                   + bcast_actx_ary_types)}):
                                if __debug__:
                                    if isinstance(arg1,
                                            {tup_str(bcast_actx_ary_types)}):
                                        warn("Broadcasting {cls} over array "
                                            f"context array type {{type(arg1)}} "
                                            "is deprecated "
                                            "and will no longer work in 2025."
                                            "There is no replacement as of right now. "
                                            "See the discussion in "
                                            "https://github.com/inducer/arraycontext/"
                                            "pull/190. ",
                                            DeprecationWarning, stacklevel=2)

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
