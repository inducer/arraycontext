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


# {{{ with_container_arithmetic

class _OpClass(enum.Enum):
    ARITHMETIC = enum.auto
    BITWISE = enum.auto
    SHIFT = enum.auto
    EQ_COMPARISON = enum.auto
    REL_COMPARISON = enum.auto


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


def _format_unary_op_str(op_str, arg1):
    if isinstance(arg1, tuple):
        arg1_entry, arg1_container = arg1
        return (f"{op_str.format(arg1_entry)} "
                f"for {arg1_entry} in {arg1_container}")
    else:
        return op_str.format(arg1)


def _format_binary_op_str(op_str, arg1, arg2):
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


def with_container_arithmetic(
        bcast_number=True, bcast_obj_array=None, bcast_numpy_array=False,
        arithmetic=True, bitwise=False, shift=False,
        eq_comparison=None, rel_comparison=None):
    """A class decorator that implements built-in operators for array containers
    by propagating the operations to the elements of the container.

    :arg bcast_number: If *True*, numbers broadcast over the container
        (with the container as the 'outer' structure).
    :arg bcast_obj_array: If *True*, :mod:`numpy` object arrays broadcast over
        the container.  (with the container as the 'inner' structure)
    :arg bcast_numpy_array: If *True*, any :class:`numpy.ndarray` will broadcast
        over the container.  (with the container as the 'inner' structure)
    :arg arithmetic: Implement the conventional arithmetic operators, including
        ``**``, :func:`divmod`, and ``//``. Also includes ``+`` and ``-`` as well as
        :func:`abs`.
    :arg bitwise: If *True*, implement bitwise and, or, not, and inversion.
    :arg shift: If *True*, implement bit shifts.
    :arg eq_comparison: If *True*, implement ``==`` and ``!=``.
    :arg rel_comparison: If *True*, implement ``<``, ``<=``, ``>``, ``>=``.
        In that case, if *eq_comparison* is unspecified, it is also set to
        *True*.

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

    if rel_comparison and eq_comparison is None:
        eq_comparison = True

    if eq_comparison is None:
        raise TypeError("eq_comparison must be specified")

    if not bcast_obj_array and bcast_numpy_array:
        raise TypeError("bcast_obj_array must be set if bcast_numpy_array is")

    if bcast_numpy_array:
        def numpy_pred(name):
            return f"isinstance({name}, np.ndarray)"
    elif bcast_obj_array:
        def numpy_pred(name):
            return f"isinstance({name}, np.ndarray) and {name}.dtype.char == 'O'"
    else:
        def numpy_pred(name):
            return "False"  # optimized away

    desired_op_classes = set()
    if arithmetic:
        desired_op_classes.add(_OpClass.ARITHMETIC)
    if bitwise:
        desired_op_classes.add(_OpClass.BITWISE)
    if shift:
        desired_op_classes.add(_OpClass.SHIFT)
    if eq_comparison:
        desired_op_classes.add(_OpClass.EQ_COMPARISON)
    if rel_comparison:
        desired_op_classes.add(_OpClass.REL_COMPARISON)

    # }}}

    def wrap(cls):
        if (not hasattr(cls, "_serialize_init_arrays_code")
                or not hasattr(cls, "_deserialize_init_arrays_code")):
            raise TypeError(f"class '{cls.__name__}' must provide serialization "
                    "code to generate arithmetic operations by implementing "
                    "'_serialize_init_arrays_code' and "
                    "'_deserialize_init_arrays_code'. If this is a dataclass, "
                    "use the 'dataclass_array_container' decorator first.")

        from pytools.codegen import CodeGenerator
        gen = CodeGenerator()
        gen("""
            from numbers import Number
            import numpy as np
            from arraycontext import ArrayContainer
            """)
        gen("")

        def same_key(k1, k2):
            assert k1 == k2
            return k1

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
            if op_cls not in desired_op_classes:
                continue

            # {{{ "forward" binary operators

            fname = f"_{cls.__name__.lower()}_{dunder_name}"
            zip_init_args = cls._deserialize_init_arrays_code("arg1", {
                    same_key(key_arg1, key_arg2):
                    _format_binary_op_str(op_str, expr_arg1, expr_arg2)
                    for (key_arg1, expr_arg1), (key_arg2, expr_arg2) in zip(
                        cls._serialize_init_arrays_code("arg1").items(),
                        cls._serialize_init_arrays_code("arg2").items())
                    })
            bcast_init_args = cls._deserialize_init_arrays_code("arg1", {
                    key_arg1: _format_binary_op_str(op_str, expr_arg1, "arg2")
                    for key_arg1, expr_arg1 in
                    cls._serialize_init_arrays_code("arg1").items()
                    })

            gen(f"""
                def {fname}(arg1, arg2):
                    if arg2.__class__ is cls:
                        return cls({zip_init_args})
                    if {bcast_number}:  # optimized away
                        if isinstance(arg2, Number):
                            return cls({bcast_init_args})
                    if {numpy_pred("arg2")}:
                        result = np.empty_like(arg2, dtype=object)
                        for i in np.ndindex(arg2.shape):
                            result[i] = {op_str.format("arg1", "arg2[i]")}
                        return result
                    return NotImplemented
                cls.__{dunder_name}__ = {fname}""")
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
                gen(f"""
                    def {fname}(arg2, arg1):
                        # assert other.__cls__ is not cls

                        if {bcast_number}:  # optimized away
                            if isinstance(arg1, Number):
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
        exec(compile(code, "<generated code>", "exec"), result_dict)

        return cls

    # we're being called as @with_container_arithmetic(...), with parens
    return wrap

# }}}


# vim: foldmethod=marker
