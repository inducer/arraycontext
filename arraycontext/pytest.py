"""
.. currentmodule:: arraycontext

.. autoclass:: PytestArrayContextFactory
.. autoclass:: PytestPyOpenCLArrayContextFactory

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

from typing import Any, Callable, Dict, Sequence, Type, Union

from arraycontext import NumpyArrayContext
from arraycontext.context import ArrayContext


# {{{ array context factories

class PytestArrayContextFactory:
    @classmethod
    def is_available(cls) -> bool:
        return True

    def __call__(self) -> ArrayContext:
        raise NotImplementedError


class PytestPyOpenCLArrayContextFactory(PytestArrayContextFactory):
    """
    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, device):
        """
        :arg device: a :class:`pyopencl.Device`.
        """
        self.device = device

    @classmethod
    def is_available(cls) -> bool:
        try:
            import pyopencl  # noqa: F401
            return True
        except ImportError:
            return False

    def get_command_queue(self):
        # Get rid of leftovers from past tests.
        # CL implementations are surprisingly limited in how many
        # simultaneous contexts they allow...
        from pyopencl.tools import clear_first_arg_caches
        clear_first_arg_caches()

        from gc import collect
        collect()

        import pyopencl as cl

        # On Intel CPU CL, existence of a command queue does not ensure that
        # the context survives.
        ctx = cl.Context([self.device])
        return ctx, cl.CommandQueue(ctx)


class _PytestPyOpenCLArrayContextFactoryWithClass(PytestPyOpenCLArrayContextFactory):
    force_device_scalars = True

    @property
    def actx_class(self):
        from arraycontext import PyOpenCLArrayContext
        return PyOpenCLArrayContext

    def __call__(self):
        # The ostensibly pointless assignment to *ctx* keeps the CL context alive
        # long enough to create the array context, which will then start
        # holding a reference to the context to keep it alive in turn.
        # On some implementations (notably Intel CPU), holding a reference
        # to a queue does not keep the context alive.
        _ctx, queue = self.get_command_queue()

        alloc = None

        if queue.device.platform.name == "NVIDIA CUDA":
            from pyopencl.tools import ImmediateAllocator
            alloc = ImmediateAllocator(queue)

            from warnings import warn
            warn("Disabling SVM due to memory leak "
                 "in Nvidia CL when running pytest. "
                 "See https://github.com/inducer/arraycontext/issues/196",
                 stacklevel=1)

        return self.actx_class(
                queue,
                allocator=alloc,
                force_device_scalars=self.force_device_scalars)

    def __str__(self):
        return (f"<{self.actx_class.__name__} "
            f"for <pyopencl.Device '{self.device.name.strip()}' "
            f"on '{self.device.platform.name.strip()}'>>")


class _PytestPyOpenCLArrayContextFactoryWithClassAndHostScalars(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    force_device_scalars = False


class _PytestPytatoPyOpenCLArrayContextFactory(PytestPyOpenCLArrayContextFactory):
    @classmethod
    def is_available(cls) -> bool:
        try:
            import pyopencl  # noqa: F401
            import pytato  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def actx_class(self):
        from arraycontext import PytatoPyOpenCLArrayContext
        actx_cls = PytatoPyOpenCLArrayContext
        return actx_cls

    def __call__(self):
        # The ostensibly pointless assignment to *ctx* keeps the CL context alive
        # long enough to create the array context, which will then start
        # holding a reference to the context to keep it alive in turn.
        # On some implementations (notably Intel CPU), holding a reference
        # to a queue does not keep the context alive.
        _ctx, queue = self.get_command_queue()

        alloc = None

        if queue.device.platform.name == "NVIDIA CUDA":
            from pyopencl.tools import ImmediateAllocator
            alloc = ImmediateAllocator(queue)

            from warnings import warn
            warn("Disabling SVM due to memory leak "
                 "in Nvidia CL when running pytest. "
                 "See https://github.com/inducer/arraycontext/issues/196",
                 stacklevel=1)

        return self.actx_class(queue, allocator=alloc)

    def __str__(self):
        return ("<PytatoPyOpenCLArrayContext for "
                f"<pyopencl.Device '{self.device.name.strip()}' "
                f"on '{self.device.platform.name.strip()}'>>")


class _PytestEagerJaxArrayContextFactory(PytestArrayContextFactory):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def is_available(cls) -> bool:
        try:
            import jax  # noqa: F401
            return True
        except ImportError:
            return False

    def __call__(self):
        from jax import config

        from arraycontext import EagerJAXArrayContext
        config.update("jax_enable_x64", True)
        return EagerJAXArrayContext()

    def __str__(self):
        return "<EagerJAXArrayContext>"


class _PytestPytatoJaxArrayContextFactory(PytestArrayContextFactory):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def is_available(cls) -> bool:
        try:
            import jax  # noqa: F401
            import pytato  # noqa: F401
            return True
        except ImportError:
            return False

    def __call__(self):
        from jax import config

        from arraycontext import PytatoJAXArrayContext
        config.update("jax_enable_x64", True)
        return PytatoJAXArrayContext()

    def __str__(self):
        return "<PytatoJAXArrayContext>"


# {{{ _PytestArrayContextFactory

class _NumpyArrayContextForTests(NumpyArrayContext):
    def transform_loopy_program(self, t_unit):
        return t_unit


class _PytestNumpyArrayContextFactory(PytestArrayContextFactory):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        return _NumpyArrayContextForTests()

    def __str__(self):
        return "<NumpyArrayContext>"

# }}}


_ARRAY_CONTEXT_FACTORY_REGISTRY: \
        Dict[str, Type[PytestArrayContextFactory]] = {
                "pyopencl": _PytestPyOpenCLArrayContextFactoryWithClass,
                "pyopencl-deprecated":
                _PytestPyOpenCLArrayContextFactoryWithClassAndHostScalars,
                "pytato:pyopencl": _PytestPytatoPyOpenCLArrayContextFactory,
                "pytato:jax": _PytestPytatoJaxArrayContextFactory,
                "eagerjax": _PytestEagerJaxArrayContextFactory,
                "numpy": _PytestNumpyArrayContextFactory,
                }


def register_pytest_array_context_factory(
        name: str,
        factory: Type[PytestArrayContextFactory]) -> None:
    if name in _ARRAY_CONTEXT_FACTORY_REGISTRY:
        raise ValueError(f"factory '{name}' already exists")

    _ARRAY_CONTEXT_FACTORY_REGISTRY[name] = factory

# }}}


# {{{ pytest integration

def pytest_generate_tests_for_array_contexts(
        factories: Sequence[Union[str, Type[PytestArrayContextFactory]]], *,
        factory_arg_name: str = "actx_factory",
        ) -> Callable[[Any], None]:
    """Parametrize tests for pytest to use an :class:`~arraycontext.ArrayContext`.

    Using this function in :mod:`pytest` test scripts allows you to use the
    argument *factory_arg_name*, which is a callable that returns a
    :class:`~arraycontext.ArrayContext`. All test functions will automatically
    be run once for each implemented array context. To select specific array
    context implementations explicitly define, for example,

    .. code-block:: python

        pytest_generate_tests = pytest_generate_tests_for_array_context([
            "pyopencl",
            ])

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
    * ``"pytato-pyopencl"``, which creates a
      :class:`~arraycontext.PytatoPyOpenCLArrayContext`.

    :arg factories: a list of identifiers or
        :class:`PytestPyOpenCLArrayContextFactory` classes (not instances)
        for which to generate test fixtures.
    """

    # {{{ get all requested array context factories

    import os
    env_factory_string = os.environ.get("ARRAYCONTEXT_TEST", None)

    if env_factory_string is not None:
        unique_factories = set(env_factory_string.split(","))
    else:
        unique_factories = set(factories)               # type: ignore[arg-type]

    if not unique_factories:
        raise ValueError("no array context factories were selected")

    unknown_factories = [
            factory for factory in unique_factories
            if (isinstance(factory, str)
                and factory not in _ARRAY_CONTEXT_FACTORY_REGISTRY)
            ]

    if unknown_factories:
        if env_factory_string is not None:
            raise RuntimeError(
                    "unknown array context factories passed through environment "
                    f"variable 'ARRAYCONTEXT_TEST': {unknown_factories}")
        else:
            raise ValueError(f"unknown array contexts: {unknown_factories}")

    available_factories = {
        factory for key in unique_factories
        for factory in [_ARRAY_CONTEXT_FACTORY_REGISTRY.get(key, key)]
        if (
            not isinstance(factory, str)
            and issubclass(factory, PytestArrayContextFactory)
            and factory.is_available())
        }

    from pytools import partition
    pyopencl_factories, other_factories = partition(
        lambda factory: issubclass(factory, PytestPyOpenCLArrayContextFactory),
        available_factories)

    # }}}

    def inner(metafunc):
        # {{{ get pyopencl devices

        import pyopencl.tools as cl_tools
        arg_names = cl_tools.get_pyopencl_fixture_arg_names(
                metafunc, extra_arg_names=[factory_arg_name])

        if not arg_names:
            return

        arg_values, ids = cl_tools.get_pyopencl_fixture_arg_values()
        empty_arg_dict = dict.fromkeys(arg_values[0])

        # }}}

        # {{{ add array context factory to arguments

        if factory_arg_name in arg_names:
            if "ctx_factory" in arg_names or "ctx_getter" in arg_names:
                raise RuntimeError(
                        f"Cannot use both an '{factory_arg_name}' and a "
                        "'ctx_factory' / 'ctx_getter' as arguments.")

            arg_values_with_actx = []

            if pyopencl_factories:
                for arg_dict in arg_values:
                    arg_values_with_actx.extend([
                        {factory_arg_name: factory(arg_dict["device"]), **arg_dict}
                        for factory in pyopencl_factories
                        ])

            if other_factories:
                arg_values_with_actx.extend([
                    {factory_arg_name: factory(), **empty_arg_dict}
                    for factory in other_factories
                    ])
        else:
            arg_values_with_actx = arg_values

        # }}}

        # NOTE: sorts the args so that parallel pytest works
        arg_value_tuples = sorted([
                tuple([arg_dict[name] for name in arg_names])
                for arg_dict in arg_values_with_actx
                ], key=lambda x: str(x))

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

    from warnings import warn
    warn("pytest_generate_tests_for_pyopencl_array_context is deprecated. "
            "Use 'pytest_generate_tests = "
            "arraycontext.pytest_generate_tests_for_array_contexts"
            "([\"pyopencl-deprecated\"])' instead. "
            "pytest_generate_tests_for_pyopencl_array_context will stop working "
            "in 2022.",
            DeprecationWarning, stacklevel=2)

    pytest_generate_tests_for_array_contexts([
        "pyopencl-deprecated",
        ], factory_arg_name="actx_factory")(metafunc)

# }}}


# vim: foldmethod=marker
