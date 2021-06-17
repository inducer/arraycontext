"""
.. currentmodule:: arraycontext
.. autofunction:: pytest_generate_tests_for_array_contexts
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


# {{{ pytest integration

from arraycontext.impl.pyopencl import PyOpenCLArrayContext
from arraycontext.impl.pytato import PytatoArrayContext
import pyopencl as cl
from pyopencl.tools import _ContextFactory


class _PyOpenCLArrayContextFactory(_ContextFactory):
    def __call__(self):
        ctx = super().__call__()
        from arraycontext.impl.pyopencl import PyOpenCLArrayContext
        return PyOpenCLArrayContext(cl.CommandQueue(ctx))

    def __str__(self):
        return ("<PyOpenCL array context factory for <pyopencl.Device '%s' on '%s'>"
                % (self.device.name.strip(),
                 self.device.platform.name.strip()))


class _PytatoArrayContextFactory(_ContextFactory):
    def __call__(self):
        ctx = super().__call__()
        from arraycontext.impl.pytato import PytatoArrayContext
        return PytatoArrayContext(cl.CommandQueue(ctx))

    def __str__(self):
        return ("<Pytato array context factory for <pyopencl.Device '%s' on '%s'>"
                % (self.device.name.strip(),
                 self.device.platform.name.strip()))


types_to_factories = {PyOpenCLArrayContext: _PyOpenCLArrayContextFactory,
    PytatoArrayContext: _PytatoArrayContextFactory}


def pytest_generate_tests_for_array_contexts(metafunc, actx_list=None) -> None:
    """Parametrize tests for pytest to use a
    :class:`~arraycontext.ArrayContext`.

    Performs device enumeration analogously to
    :func:`pyopencl.tools.pytest_generate_tests_for_pyopencl`.

    Using the line:

    .. code-block:: python

       from arraycontext import pytest_generate_tests_for_array_contexts
            as pytest_generate_tests

    in your pytest test scripts allows you to use the argument ``actx_factory``,
    in your test functions, and they will automatically be
    run once for each OpenCL device/platform in the system, as appropriate,
    with an argument-less function that returns an
    :class:`~arraycontext.ArrayContext` when called.

    It also allows you to specify the ``PYOPENCL_TEST`` environment variable
    for device selection.
    """

    if actx_list is None:
        actx_list = [PyOpenCLArrayContext, PytatoArrayContext]

    actx_factories = [types_to_factories[a] for a in actx_list]

    import pyopencl.tools as cl_tools
    arg_names = cl_tools.get_pyopencl_fixture_arg_names(
            metafunc, extra_arg_names=["actx_factory"])

    if not arg_names:
        return

    arg_values, ids = cl_tools.get_pyopencl_fixture_arg_values()
    if "actx_factory" in arg_names:
        if "ctx_factory" in arg_names or "ctx_getter" in arg_names:
            raise RuntimeError("Cannot use both an 'actx_factory' and a "
                    "'ctx_factory' / 'ctx_getter' as arguments.")

        for arg_dict in arg_values:
            dev = arg_dict["device"]
            extra_factories = []

            for factory in actx_factories:
                if "actx_factory" in arg_dict:
                    arg_dict["actx_factory_"+str(factory)] = factory(dev)
                    extra_factories += ("actx_factory_"+str(factory),)
                else:
                    arg_dict["actx_factory"] = factory(dev)

    arg_values_out = [
            tuple(arg_dict[name] for name in arg_names)
            for arg_dict in arg_values
            ]

    for extra_factory in extra_factories:
        arg_values_out += [
                tuple((arg_dict[extra_factory],))
                for arg_dict in arg_values
                ]

    metafunc.parametrize(arg_names, arg_values_out, ids=ids)


def pytest_generate_tests_for_pyopencl_array_context(metafunc) -> None:
    from warnings import warn
    warn("'pytato.pytest_generate_tests_for_pyopencl_array_context' "
         "is deprecated, use 'pytato.pytest_generate_tests_for_array_contexts' "
         "instead. Only tests with PyOpenCLArrayContext will be run.",
         DeprecationWarning)
    pytest_generate_tests_for_array_contexts(metafunc,
                                            actx_list=[PyOpenCLArrayContext])

# }}}


# vim: foldmethod=marker
