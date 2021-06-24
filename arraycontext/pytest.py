"""
.. currentmodule:: arraycontext

.. autoclass:: PytestArrayContextFactory

.. autofunction:: pytest_generate_tests_for_array_contexts
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

from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

import pyopencl as cl
from arraycontext.context import ArrayContext


# {{{ array context factories

class PytestArrayContextFactory:
    """
    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, device):
        """
        :arg device: a :class:`pyopencl.Device`.
        """
        self.device = device

    def get_command_queue(self):
        # Get rid of leftovers from past tests.
        # CL implementations are surprisingly limited in how many
        # simultaneous contexts they allow...
        from pyopencl.tools import clear_first_arg_caches
        clear_first_arg_caches()

        from gc import collect
        collect()

        ctx = cl.Context([self.device])
        return cl.CommandQueue(ctx)

    def __call__(self) -> ArrayContext:
        raise NotImplementedError


class _PyOpenCLArrayContextFactory(PytestArrayContextFactory):
    force_device_scalars = True

    def __call__(self):
        from arraycontext import PyOpenCLArrayContext
        return PyOpenCLArrayContext(
                self.get_command_queue(),
                force_device_scalars=self.force_device_scalars)

    def __str__(self):
        return ("<PyOpenCLArrayContext for <pyopencl.Device '%s' on '%s'>" %
                (self.device.name.strip(),
                 self.device.platform.name.strip()))


class _DeprecatedPyOpenCLArrayContextFactory(_PyOpenCLArrayContextFactory):
    force_device_scalars = False


_ARRAY_CONTEXT_FACTORY_DICT: Dict[str, Type[PytestArrayContextFactory]] = {
        "pyopencl": _PyOpenCLArrayContextFactory,
        }
_ALL_ARRAY_CONTEXT_FACTORY_DICT: Dict[str, Type[PytestArrayContextFactory]] = {
        "pyopencl-deprecated": _DeprecatedPyOpenCLArrayContextFactory,
        }
_ALL_ARRAY_CONTEXT_FACTORY_DICT.update(_ARRAY_CONTEXT_FACTORY_DICT)


def register_array_context_factory(
        name: str,
        factory: Type[PytestArrayContextFactory]) -> None:
    if name in _ALL_ARRAY_CONTEXT_FACTORY_DICT:
        raise ValueError(f"factory '{name}' already exists")

    _ARRAY_CONTEXT_FACTORY_DICT[name] = factory
    _ALL_ARRAY_CONTEXT_FACTORY_DICT[name] = factory

# }}}


# {{{ pytest integration

def pytest_generate_tests_for_array_contexts(
        factories: Optional[Sequence[
            Union[str, Type[PytestArrayContextFactory]]
            ]] = None,
        ) -> Callable[[Any], None]:
    """Parametrize tests for pytest to use an :class:`~arraycontext.ArrayContext`.

    Using this function in :mod:`pytest` test scripts allows you to use the
    argument ``actx_factory``, which is a callable that returns a
    :class:`~arraycontext.ArrayContext`. All test functions will automatically
    be run once for each implemented array context. To select specific array
    context implementations explicitly define, for example,

    .. code-block:: python

        pytest_generate_tests = \
            pytest_generate_tests_for_array_context(["pyopencl"])

    to use the :mod:`pyopencl`-based array context. For :mod:`pyopencl`-based
    contexts :func:`pyopencl.tools.pytest_generate_tests_for_pyopencl` is used
    as a backend, which allows specifying the ``PYOPENCL_TEST`` environment
    variable for device selection.

    The environment variable ``ARRAYCONTEXT_TEST`` can also be used to
    overwrite any chosen implementations through *factories*. This is a
    comma-separated list of known array contexts.

    Current supported implementations include:

    * ``"pyopencl"``, which creates a :class:`~arraycontext.PyOpenCLArrayContext`
      with ``force_device_scalars=True``.
    * ``"pyopencl-deprecated"``, which creates a
      :class:`~arraycontext.PyOpenCLArrayContext` with
      ``force_device_scalars=False``.

    :arg factories: a list of identifiers or
        :class:`PytestArrayContextFactory` classes (not instances).
    """

    # {{{ get all requested array context factories

    import os
    env_factory_string = os.environ.get("ARRAYCONTEXT_TEST", None)

    if env_factory_string is not None:
        unique_factories = set(env_factory_string.split(","))

        unknown_factories = [
                factory for factory in unique_factories
                if (isinstance(factory, str)
                    and factory not in _ALL_ARRAY_CONTEXT_FACTORY_DICT)
                ]
        if unknown_factories:
            raise RuntimeError(
                    "unknown array context factories passed through environment "
                    f"variable 'ARRAYCONTEXT_TEST': {unknown_factories}")
    else:
        if factories is None:
            unique_factories = set(
                    _ARRAY_CONTEXT_FACTORY_DICT.values())   # type: ignore[arg-type]
        else:
            unique_factories = set(factories)               # type: ignore[arg-type]
            unknown_factories = [
                    factory for factory in unique_factories
                    if (isinstance(factory, str)
                        and factory not in _ALL_ARRAY_CONTEXT_FACTORY_DICT)
                    ]
            if unknown_factories:
                raise ValueError(f"unknown array contexts: {unknown_factories}")

    if not unique_factories:
        raise ValueError("no array context factories were selected")

    unique_factories = set([
        _ALL_ARRAY_CONTEXT_FACTORY_DICT.get(factory, factory)  # type: ignore[misc]
        for factory in unique_factories])

    # }}}

    def inner(metafunc):
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
                for factory in unique_factories:
                    arg_values_with_actx.append(dict(
                        actx_factory=factory(arg_dict["device"]),
                        **arg_dict))
        else:
            arg_values_with_actx = arg_values

        arg_value_tuples = [
                tuple(arg_dict[name] for name in arg_names)
                for arg_dict in arg_values_with_actx
                ]

        # }}}

        metafunc.parametrize(arg_names, arg_value_tuples, ids=ids)

    return inner


def pytest_generate_tests_for_pyopencl_array_context(metafunc) -> None:
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
    pytest_generate_tests_for_array_contexts(["pyopencl-deprecated"])(metafunc)

# }}}


# vim: foldmethod=marker
