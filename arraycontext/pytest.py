"""
.. currentmodule:: arraycontext

.. autofunction:: pytest_generate_tests_for_array_context
.. autofunction:: pytest_generate_tests_for_pyopencl_array_context
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

from typing import Optional, Sequence

import pyopencl as cl
from pyopencl.tools import _ContextFactory


# {{{ array context factories

class _PyOpenCLArrayContextFactory(_ContextFactory):
    force_device_scalars = True

    def __call__(self):
        ctx = super().__call__()
        from arraycontext.impl.pyopencl import PyOpenCLArrayContext
        return PyOpenCLArrayContext(
                cl.CommandQueue(ctx),
                force_device_scalars=self.force_device_scalars)

    def __str__(self):
        return ("<array context factory for <pyopencl.Device '%s' on '%s'>" %
                (self.device.name.strip(),
                 self.device.platform.name.strip()))


class _DeprecatedPyOpenCLArrayContextFactory(_PyOpenCLArrayContextFactory):
    force_device_scalars = False


_ARRAY_CONTEXT_FACTORY_DICT = {
        "pyopencl": _PyOpenCLArrayContextFactory,
        }
_ALL_ARRAY_CONTEXT_FACTORY_DICT = {
        "pyopencl-deprecated": _DeprecatedPyOpenCLArrayContextFactory,
        }

_ALL_ARRAY_CONTEXT_FACTORY_DICT.update(_ARRAY_CONTEXT_FACTORY_DICT)

# }}}


# {{{ pytest integration

def pytest_generate_tests_for_array_context(
        metafunc,
        impls: Optional[Sequence[str]] = None):
    """Parametrize tests for pytest to use an :class:`~arraycontext.ArrayContext`.

    Using the line

    .. code-block:: python

       from arraycontext import (
            pytest_generate_tests_for_array_context
            as pytest_generate_tests)

    in your pytest test scripts allows you to use the argument ``actx_factory``,
    which is a callable that returns a :class:`~arraycontext.ArrayContext`.
    All test functions will automaticall be run once for each implemented array
    context. To select specific array context implementations explicitly
    define, for example,

    .. code-block:: python

        def pytest_generate_tests(metafunc):
            pytest_generate_tests_for_array_context(metafunc, impls=["pyopencl"])

    to use the :mod:`pyopencl`-based array context. For :mod:`pyopencl`-based
    contexts :func:`pyopencl.tools.pytest_generate_tests_for_pyopencl` is used
    as a backend, which allows specifying the ``PYOPENCL_TEST`` environment
    variable for device selection.

    The environment variable ``ARRAYCONTEXT_TEST`` can also be used to
    overwrite any chosen implementations through *impls*.

    Current supported implementations include:

    * ``"pyopencl"``, which creates a :class:`~arraycontext.PyOpenCLArrayContext`
      with ``force_device_scalars=True``.
    * ``"pyopencl-deprecated"``, which creates a
      :class:`~arraycontext.PyOpenCLArrayContext` with
      ``force_device_scalars=False``.

    :arg impls: a list of identifiers for desired implementations.
    """

    # {{{ get all requested array context factories

    import os
    env_impls_string = os.environ.get("ARRAYCONTEXT_TEST", None)

    if env_impls_string is not None:
        impls = set(env_impls_string.split(","))

        unknown_impls = [
                impl for impl in impls
                if impl not in _ALL_ARRAY_CONTEXT_FACTORY_DICT]
        if unknown_impls:
            raise RuntimeError(
                    "unknown array contexts passed through environment "
                    f"variable 'ARRAYCONTEXT_TEST': {unknown_impls}")
    else:
        if impls is None:
            impls = set(_ARRAY_CONTEXT_FACTORY_DICT.values())
        else:
            impls = set(impls)
            unknown_impls = [
                    impl for impl in impls
                    if impl not in _ALL_ARRAY_CONTEXT_FACTORY_DICT]
            if unknown_impls:
                raise ValueError(f"unknown array contexts: {unknown_impls}")

    # }}}

    # {{{ get pyopencl devices

    import pyopencl.tools as cl_tools
    arg_names = cl_tools.get_pyopencl_fixture_arg_names(
            metafunc, extra_arg_names=["actx_factory"])

    if not arg_names:
        return

    arg_values, ids = cl_tools.get_pyopencl_fixture_arg_values()

    # }}}

    # {{{ add array context factory to arguments

    if "actx_factory" in arg_names:
        if "ctx_factory" in arg_names or "ctx_getter" in arg_names:
            raise RuntimeError("Cannot use both an 'actx_factory' and a "
                    "'ctx_factory' / 'ctx_getter' as arguments.")

        arg_values_with_actx = []
        for arg_dict in arg_values:
            for impl in impls:
                arg = arg_dict.copy()
                arg["actx_factory"] = \
                        _ALL_ARRAY_CONTEXT_FACTORY_DICT[impl](arg_dict["device"])

                arg_values_with_actx.append(arg)
    else:
        arg_values_with_actx = arg_values

    arg_value_tuples = [
            tuple(arg_dict[name] for name in arg_names)
            for arg_dict in arg_values_with_actx
            ]

    # }}}

    metafunc.parametrize(arg_names, arg_value_tuples, ids=ids)


def pytest_generate_tests_for_pyopencl_array_context(metafunc):
    """Parametrize tests for pytest to use a
    :class:`~arraycontext.PyOpenCLArrayContext`.

    Performs device enumeration analogously to
    :func:`pyopencl.tools.pytest_generate_tests_for_pyopencl`.

    Using the line:

    .. code-block:: python

       from arraycontext import (
            pytest_generate_tests_for_pyopencl_array_context
            as pytest_generate_tests)

    in your pytest test scripts allows you to use the argument ``actx_factory``,
    in your test functions, and they will automatically be
    run once for each OpenCL device/platform in the system, as appropriate,
    with an argument-less function that returns an
    :class:`~arraycontext.ArrayContext` when called.

    It also allows you to specify the ``PYOPENCL_TEST`` environment variable
    for device selection.
    """
    pytest_generate_tests_for_array_context(metafunc, impls=["pyopencl-deprecated"])

# }}}


# vim: foldmethod=marker
