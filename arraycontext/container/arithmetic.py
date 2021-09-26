# mypy: disallow-untyped-defs

"""
.. currentmodule:: arraycontext
.. autofunction:: with_container_arithmetic
"""

import enum


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

from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

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
        arg1: Union[Tuple[str, ...], str],
        arg2: Union[Tuple[str, ...], str]) -> str:
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


class _FailSafe:
    pass


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
    :arg _bcast_actx_array_type: If *True*, instances of base array types of the
        container's array context are broadcasted over the container. Can be
        *True* only if the container has *_cls_has_array_context_attr* set.
        Defaulted to *bcast_number* if *_cls_has_array_context_attr* is set,
        else *False*.
    :arg bcast_obj_array: If *True*, :mod:`numpy` object arrays broadcast over
        the container.  (with the container as the 'inner' structure)
    :arg bcast_numpy_array: If *True*, any :class:`numpy.ndarray` will broadcast
        over the container.  (with the container as the 'inner' structure)
        If this is set to *True*, *bcast_obj_array* must also be *True*.
    :arg bcast_container_types: A sequence of container types that will broadcast
        over this container (with this container as the 'outer' structure).
        :class:`numpy.ndarray` is permitted to be part of this sequence to
        indicate that, in such broadcasting situations, this container should
        be the 'outer' structure. In this case, *bcast_obj_array*
        (and consequently *bcast_numpy_array*) must be *False*.
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

    # {{{ handle inputs

    if bcast_obj_array is None:
        raise TypeError("bcast_obj_array must be specified")

    if rel_comparison is None:
        raise TypeError("rel_comparison must be specified")

    if bcast_numpy_array:
        from warnings import warn
        warn("'bcast_numpy_array=True' is deprecated and will be unsupported"
             " from December 2021", DeprecationWarning, stacklevel=2)

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
            return f"isinstance({name}, np.ndarray)"
    elif bcast_obj_array:
        def numpy_pred(name: str) -> str:
            return f"isinstance({name}, np.ndarray) and {name}.dtype.char == 'O'"
    else:
        def numpy_pred(name: str) -> str:
            return "False"  # optimized away

    if bcast_container_types is None:
        bcast_container_types = ()
    bcast_container_types_count = len(bcast_container_types)

    if np.ndarray in bcast_container_types and bcast_obj_array:
        raise ValueError("If numpy.ndarray is part of bcast_container_types, "
                "bcast_obj_array must be False.")

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
        cls_has_array_context_attr: Optional[Union[bool, Type[_FailSafe]]] = \
                _cls_has_array_context_attr
        bcast_actx_array_type: Optional[Union[bool, Type[_FailSafe]]] = \
                _bcast_actx_array_type

        if cls_has_array_context_attr is None:
            if hasattr(cls, "array_context"):
                cls_has_array_context_attr = _FailSafe
                warn(f"{cls} has an 'array_context' attribute, but it does not "
                        "set '_cls_has_array_context_attr' to 'True' when calling "
                        "'with_container_arithmetic'. This is being interpreted "
                        "as 'array_context' being permitted to fail. Tolerating "
                        "these failures comes at a substantial cost. It is "
                        "deprecated and will stop working in 2023. "
                        "Having a working 'array_context' attribute is desirable "
                        "to enable arithmetic with other array types supported "
                        "by the array context. "
                        f"If '{cls.__name__}.array_context' will not fail, pass "
                        "'_cls_has_array_context_attr=True'. "
                        "If you do not want container arithmetic to make "
                        "use of the array context, set "
                        "'_cls_has_array_context_attr=False'.",
                        stacklevel=2)

        if bcast_actx_array_type is None:
            if cls_has_array_context_attr:
                if bcast_number:
                    # copy over _FailSafe if present
                    bcast_actx_array_type = cls_has_array_context_attr
            else:
                bcast_actx_array_type = False
        else:
            if bcast_actx_array_type and not cls_has_array_context_attr:
                raise TypeError("_bcast_actx_array_type can be True only if "
                                "_cls_has_array_context_attr is set.")

        if (not hasattr(cls, "_serialize_init_arrays_code")
                or not hasattr(cls, "_deserialize_init_arrays_code")):
            raise TypeError(f"class '{cls.__name__}' must provide serialization "
                    "code to generate arithmetic operations by implementing "
                    "'_serialize_init_arrays_code' and "
                    "'_deserialize_init_arrays_code'. If this is a dataclass, "
                    "use the 'dataclass_array_container' decorator first.")

        if cls_has_array_context_attr is _FailSafe:
            def actx_getter_code(arg: str) -> str:
                return f"_get_actx({arg})"
        else:
            def actx_getter_code(arg: str) -> str:
                return f"{arg}.array_context"

        from pytools.codegen import CodeGenerator, Indentation
        gen = CodeGenerator()
        gen("""
            from numbers import Number
            import numpy as np
            from arraycontext import (
                ArrayContainer, get_container_context_recursively)
            from warnings import warn

            def _raise_if_actx_none(actx):
                if actx is None:
                    raise ValueError("array containers with frozen arrays "
                        "cannot be operated upon")
                return actx

            def _get_actx(ary):
                try:
                    return ary.array_context
                except Exception as e:
                    warn(f"Accessing '{type(ary).__name__}.array_context' failed "
                        f"({type(e)}: {e}). This should not happen and is "
                        "deprecated. "
                        "Please fix the implementation of "
                        f"'{type(ary).__name__}.array_context' "
                        "and then set _cls_has_array_context_attr=True when "
                        "calling with_container_arithmetic to avoid the run time "
                        "cost of the check that gave you this warning. "
                        "Using expensive recovery for now.",
                        DeprecationWarning, stacklevel=3)

                return get_container_context_recursively(ary)

            def _get_actx_array_types_failsafe(ary):
                try:
                    actx = ary.array_context
                except Exception as e:
                    warn(f"Accessing '{type(ary).__name__}.array_context' failed "
                        f"({type(e)}: {e}). This should not happen and is "
                        "deprecated. "
                        "Please fix the implementation of "
                        f"'{type(ary).__name__}.array_context' "
                        "and then set _cls_has_array_context_attr=True when "
                        "calling with_container_arithmetic to avoid the run time "
                        "cost of the check that gave you this warning. "
                        "Using expensive recovery for now.",
                        DeprecationWarning, stacklevel=3)

                    actx = get_container_context_recursively(ary)

                if actx is None:
                    return ()

                return actx.array_types
            """)
        gen("")

        if bcast_container_types:
            for i, bct in enumerate(bcast_container_types):
                gen(f"from {bct.__module__} import {bct.__qualname__} as _bctype{i}")
            gen("")
        outer_bcast_type_names = tuple([
                f"_bctype{i}" for i in range(bcast_container_types_count)
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

            # {{{ "forward" binary operators

            zip_init_args = cls._deserialize_init_arrays_code("arg1", {
                    same_key(key_arg1, key_arg2):
                    _format_binary_op_str(op_str, expr_arg1, expr_arg2)
                    for (key_arg1, expr_arg1), (key_arg2, expr_arg2) in zip(
                        cls._serialize_init_arrays_code("arg1").items(),
                        cls._serialize_init_arrays_code("arg2").items())
                    })
            bcast_same_cls_init_args = cls._deserialize_init_arrays_code("arg1", {
                    key_arg1: _format_binary_op_str(op_str, expr_arg1, "arg2")
                    for key_arg1, expr_arg1 in
                    cls._serialize_init_arrays_code("arg1").items()
                    })

            gen(f"def {fname}(arg1, arg2):")
            with Indentation(gen):
                gen("if arg2.__class__ is cls:")
                with Indentation(gen):
                    if __debug__ and cls_has_array_context_attr:
                        gen(f"""
                            arg1_actx = {actx_getter_code("arg1")}
                            arg2_actx = {actx_getter_code("arg2")}
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

                if bcast_actx_array_type is _FailSafe:
                    bcast_actx_ary_types: Tuple[str, ...] = (
                        "*_get_actx_array_types_failsafe(arg1)",)
                elif bcast_actx_array_type:
                    if __debug__:
                        bcast_actx_ary_types = (
                            "*_raise_if_actx_none("
                            f"{actx_getter_code('arg1')}).array_types",)
                    else:
                        bcast_actx_ary_types = (
                                f"*{actx_getter_code('arg1')}.array_types",)
                else:
                    bcast_actx_ary_types = ()

                gen(f"""
                if {bool(outer_bcast_type_names)}:  # optimized away
                    if isinstance(arg2,
                                  {tup_str(outer_bcast_type_names
                                           + bcast_actx_ary_types)}):
                        return cls({bcast_same_cls_init_args})
                if {numpy_pred("arg2")}:
                    result = np.empty_like(arg2, dtype=object)
                    for i in np.ndindex(arg2.shape):
                        result[i] = {op_str.format("arg1", "arg2[i]")}
                    return result
                return NotImplemented
                """)
            gen(f"cls.__{dunder_name}__ = {fname}")
            gen("")

            # }}}

            # {{{ "reverse" binary operators

            if reversible:
                fname = f"_{cls.__name__.lower()}_r{dunder_name}"
                bcast_init_args = cls._deserialize_init_arrays_code("arg2", {
                        key_arg2: _format_binary_op_str(
                            op_str, "arg1", expr_arg2)
                        for key_arg2, expr_arg2 in
                        cls._serialize_init_arrays_code("arg2").items()
                        })

                if bcast_actx_array_type is _FailSafe:
                    bcast_actx_ary_types = (
                        "*_get_actx_array_types_failsafe(arg2)",)
                elif bcast_actx_array_type:
                    if __debug__:
                        bcast_actx_ary_types = (
                            "*_raise_if_actx_none("
                            f"{actx_getter_code('arg2')}).array_types",)
                    else:
                        bcast_actx_ary_types = (
                                f"*{actx_getter_code('arg2')}.array_types",)
                else:
                    bcast_actx_ary_types = ()

                gen(f"""
                    def {fname}(arg2, arg1):
                        # assert other.__cls__ is not cls

                        if {bool(outer_bcast_type_names)}:  # optimized away
                            if isinstance(arg1,
                                          {tup_str(outer_bcast_type_names
                                                   + bcast_actx_ary_types)}):
                                return cls({bcast_init_args})
                        if {numpy_pred("arg1")}:
                            result = np.empty_like(arg1, dtype=object)
                            for i in np.ndindex(arg1.shape):
                                result[i] = {op_str.format("arg1[i]", "arg2")}
                            return result
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
