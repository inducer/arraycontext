"""
Workflow:

1. If a new array context is implemented, it should be added to
   :func:`initialize_contexts`.

2. If a new function is implemented, it should be added to the
   corresponding ``write_section_name`` function.

3. Once everything is added, regenerate the tables using

.. code::

    python make_numpy_support_table.py numpy_coverage.rst
"""
from __future__ import annotations

import pathlib

from mako.template import Template

import arraycontext


# {{{ templating

HEADER = """
.. raw:: html

    <style> .red {color:red} </style>
    <style> .green {color:green} </style>

.. role:: red
.. role:: green
"""

TABLE_TEMPLATE = Template("""
${title}
${'~' * len(title)}

.. list-table::
    :header-rows: 1

    * - Function
% for ctx in contexts:
      - :class:`~arraycontext.${type(ctx).__name__}`
% endfor
% for name, (directive, in_context) in numpy_functions_for_context.items():
    * - :${directive}:`numpy.${name}`
    % for ctx in contexts:
    <%
    flag = in_context.get(type(ctx), "yes").capitalize()
    color = "green" if flag == "Yes" else "red"
    %>  - :${color}:`${flag}`
    % endfor
% endfor
""")


def initialize_contexts():
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    return [
            arraycontext.PyOpenCLArrayContext(queue, force_device_scalars=True),
            arraycontext.EagerJAXArrayContext(),
            arraycontext.PytatoPyOpenCLArrayContext(queue),
            arraycontext.PytatoJAXArrayContext(),
            ]


def build_supported_functions(funcs, contexts):
    import numpy as np

    numpy_functions_for_context = {}
    for directive, name in funcs:
        if not hasattr(np, name):
            raise ValueError(f"'{name}' not found in numpy namespace")

        in_context = {}
        for ctx in contexts:
            try:
                _ = getattr(ctx.np, name)
            except AttributeError:
                in_context[type(ctx)] = "No"

        numpy_functions_for_context[name] = (directive, in_context)

    return numpy_functions_for_context

# }}}


# {{{ writing

def write_array_creation_routines(outf, contexts):
    # https://numpy.org/doc/stable/reference/routines.array-creation.html
    funcs = (
            # (sphinx-directive, name)
            ("func", "empty_like"),
            ("func", "ones_like"),
            ("func", "zeros_like"),
            ("func", "full_like"),
            ("func", "copy"),
            )

    r = TABLE_TEMPLATE.render(
            title="Array creation routines",
            contexts=contexts,
            numpy_functions_for_context=build_supported_functions(funcs, contexts),
            )
    outf.write(r)


def write_array_manipulation_routines(outf, contexts):
    # https://numpy.org/doc/stable/reference/routines.array-manipulation.html
    funcs = (
            # (sphinx-directive, name)
            ("func", "reshape"),
            ("func", "ravel"),
            ("func", "transpose"),
            ("func", "broadcast_to"),
            ("func", "concatenate"),
            ("func", "stack"),
            )

    r = TABLE_TEMPLATE.render(
            title="Array manipulation routines",
            contexts=contexts,
            numpy_functions_for_context=build_supported_functions(funcs, contexts),
            )
    outf.write(r)


def write_linear_algebra(outf, contexts):
    # https://numpy.org/doc/stable/reference/routines.linalg.html
    funcs = (
            # (sphinx-directive, name)
            ("func", "vdot"),
            )

    r = TABLE_TEMPLATE.render(
            title="Linear algebra",
            contexts=contexts,
            numpy_functions_for_context=build_supported_functions(funcs, contexts),
            )
    outf.write(r)


def write_logic_functions(outf, contexts):
    # https://numpy.org/doc/stable/reference/routines.logic.html
    funcs = (
            # (sphinx-directive, name)
            ("func", "all"),
            ("func", "any"),
            ("data", "greater"),
            ("data", "greater_equal"),
            ("data", "less"),
            ("data", "less_equal"),
            ("data", "equal"),
            ("data", "not_equal"),
            )

    r = TABLE_TEMPLATE.render(
            title="Logic Functions",
            contexts=contexts,
            numpy_functions_for_context=build_supported_functions(funcs, contexts),
            )
    outf.write(r)


def write_mathematical_functions(outf, contexts):
    # https://numpy.org/doc/stable/reference/routines.math.html
    funcs = (
            ("data", "sin"),
            ("data", "cos"),
            ("data", "tan"),
            ("data", "arcsin"),
            ("data", "arccos"),
            ("data", "arctan"),
            ("data", "arctan2"),
            ("data", "sinh"),
            ("data", "cosh"),
            ("data", "tanh"),
            ("data", "floor"),
            ("data", "ceil"),
            ("func", "sum"),
            ("data", "exp"),
            ("data", "log"),
            ("data", "log10"),
            ("func", "real"),
            ("func", "imag"),
            ("data", "conjugate"),
            ("data", "maximum"),
            ("func", "amax"),
            ("data", "minimum"),
            ("func", "amin"),
            ("data", "sqrt"),
            ("data", "absolute"),
            ("data", "fabs"),
            )

    r = TABLE_TEMPLATE.render(
            title="Mathematical functions",
            contexts=contexts,
            numpy_functions_for_context=build_supported_functions(funcs, contexts),
            )
    outf.write(r)


def write_searching_sorting_and_counting(outf, contexts):
    # https://numpy.org/doc/stable/reference/routines.sort.html
    funcs = (
            ("func", "where"),
            )

    r = TABLE_TEMPLATE.render(
            title="Sorting, searching, and counting",
            contexts=contexts,
            numpy_functions_for_context=build_supported_functions(funcs, contexts),
            )
    outf.write(r)

# }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", type=pathlib.Path, default=None)
    args = parser.parse_args()

    def write(outf):
        outf.write(HEADER)
        write_array_creation_routines(outf, ctxs)
        write_array_manipulation_routines(outf, ctxs)
        write_linear_algebra(outf, ctxs)
        write_logic_functions(outf, ctxs)
        write_mathematical_functions(outf, ctxs)

    ctxs = initialize_contexts()

    if args.filename:
        with open(args.filename, "w") as outf:
            write(outf)
    else:
        import sys
        write(sys.stdout)
