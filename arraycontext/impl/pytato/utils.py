from __future__ import annotations


__doc__ = """
.. autofunction:: transfer_from_numpy
.. autofunction:: transfer_to_numpy


Profiling-related functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: tabulate_profiling_data

References
^^^^^^^^^^

.. autoclass:: ArrayOrNamesTc

    A constrained type variable binding to either
    :class:`pytato.Array` or :class:`pytato.AbstractResultWithNames`.
"""


__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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


from typing import TYPE_CHECKING, cast

from typing_extensions import override

import pytools
from pytato.analysis import get_num_call_sites
from pytato.array import (
    Array,
    Axis as PtAxis,
    DataInterface,
    DataWrapper,
    Placeholder,
    SizeParam,
    make_placeholder,
)
from pytato.target.loopy import LoopyPyOpenCLTarget
from pytato.transform import (
    ArrayOrNames,
    ArrayOrNamesTc,
    CopyMapper,
    TransformMapperCache,
    deduplicate,
)
from pytools import UniqueNameGenerator, memoize_method

from arraycontext.impl.pyopencl.taggable_cl_array import Axis as ClAxis


if TYPE_CHECKING:
    from collections.abc import Mapping

    import loopy as lp
    from pytato import AbstractResultWithNamedArrays
    from pytato.function import FunctionDefinition

    from arraycontext import ArrayContext
    from arraycontext.container import SerializationKey
    from arraycontext.impl.pytato import PytatoPyOpenCLArrayContext


class _DatawrapperToBoundPlaceholderMapper(CopyMapper):
    """
    Helper mapper for :func:`normalize_pt_expr`. Every
    :class:`pytato.DataWrapper` is replaced with a deterministic copy of
    :class:`Placeholder`.
    """
    def __init__(
            self,
            err_on_collision: bool = True,
            err_on_created_duplicate: bool = True,
            _cache: TransformMapperCache[ArrayOrNames, []] | None = None,
            _function_cache: TransformMapperCache[FunctionDefinition, []] | None = None
            ) -> None:
        super().__init__(
            err_on_collision=err_on_collision,
            err_on_created_duplicate=err_on_created_duplicate,
            _cache=_cache,
            _function_cache=_function_cache)

        self.bound_arguments: dict[str, DataInterface] = {}
        self.vng: UniqueNameGenerator = UniqueNameGenerator()
        self.seen_inputs: set[str] = set()

    @override
    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        if expr.name is not None:
            if expr.name in self.seen_inputs:
                raise ValueError("Got multiple inputs with the name"
                                 f"{expr.name} => Illegal.")
            self.seen_inputs.add(expr.name)

        # Normalizing names so that more arrays can have the same normalized DAG.
        from pytato.codegen import _generate_name_for_temp
        name = _generate_name_for_temp(expr, self.vng, "_actx_dw")
        self.bound_arguments[name] = expr.data
        return make_placeholder(
                    name=name,
                    shape=tuple(
                            cast("Array", self.rec(s))
                            if isinstance(s, Array) else s
                            for s in expr.shape),
                    dtype=expr.dtype,
                    axes=expr.axes,
                    tags=expr.tags)

    @override
    def map_size_param(self, expr: SizeParam) -> Array:
        raise NotImplementedError

    @override
    def map_placeholder(self, expr: Placeholder) -> Array:
        raise ValueError("Placeholders cannot appear in"
                         " DatawrapperToBoundPlaceholderMapper.")

    @override
    def map_function_definition(
            self, expr: FunctionDefinition) -> FunctionDefinition:
        raise ValueError("Function definitions cannot appear in"
                         " DatawrapperToBoundPlaceholderMapper.")


# FIXME: This strategy doesn't work if the DAG has functions, since function
# definitions can't contain non-argument placeholders
def _normalize_pt_expr(
        expr: AbstractResultWithNamedArrays
        ) -> tuple[AbstractResultWithNamedArrays,
                    Mapping[str, DataInterface]]:
    """
    Returns ``(normalized_expr, bound_arguments)``.  *normalized_expr* is a
    normalized form of *expr*, with all instances of
    :class:`pytato.DataWrapper` replaced with instances of :class:`Placeholder`
    named in a deterministic manner. The data corresponding to the placeholders
    in *normalized_expr* is recorded in the mapping *bound_arguments*.
    Deterministic naming of placeholders permits more effective caching of
    equivalent graphs.
    """
    expr = deduplicate(expr)

    if get_num_call_sites(expr):
        raise NotImplementedError(
            "_normalize_pt_expr is not compatible with expressions that "
            "contain function calls.")

    normalize_mapper = _DatawrapperToBoundPlaceholderMapper()
    normalized_expr = normalize_mapper(expr)
    return normalized_expr, normalize_mapper.bound_arguments


def get_pt_axes_from_cl_axes(axes: tuple[ClAxis, ...]) -> tuple[PtAxis, ...]:
    return tuple(PtAxis(axis.tags) for axis in axes)


def get_cl_axes_from_pt_axes(axes: tuple[PtAxis, ...]) -> tuple[ClAxis, ...]:
    return tuple(ClAxis(axis.tags) for axis in axes)


# {{{ arg-size-limiting loopy target

class ArgSizeLimitingPytatoLoopyPyOpenCLTarget(LoopyPyOpenCLTarget):
    def __init__(self, limit_arg_size_nbytes: int) -> None:
        super().__init__()
        self.limit_arg_size_nbytes: int = limit_arg_size_nbytes

    @memoize_method
    def get_loopy_target(self) -> lp.PyOpenCLTarget:
        from loopy import PyOpenCLTarget
        return PyOpenCLTarget(limit_arg_size_nbytes=self.limit_arg_size_nbytes)

# }}}


# {{{ Transfer mappers

class TransferFromNumpyMapper(CopyMapper):
    """A mapper to transfer arrays contained in :class:`~pytato.array.DataWrapper`
    instances to be device arrays, using
    :meth:`~arraycontext.ArrayContext.from_numpy`.
    """
    def __init__(self, actx: ArrayContext) -> None:
        super().__init__()
        self.actx: ArrayContext = actx

    @override
    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        import numpy as np

        if not isinstance(expr.data, np.ndarray):
            raise ValueError("TransferFromNumpyMapper: tried to transfer data that "
                             "is already on the device")

        # Ideally, this code should just do
        # return self.actx.from_numpy(expr.data).tagged(expr.tags),
        # but there seems to be no way to transfer the non_equality_tags in that case.
        actx_ary = self.actx.from_numpy(expr.data)
        assert isinstance(actx_ary, DataWrapper)

        # https://github.com/pylint-dev/pylint/issues/3893
        # pylint: disable=unexpected-keyword-arg
        return DataWrapper(
            data=actx_ary.data,
            shape=expr.shape,
            axes=expr.axes,
            tags=expr.tags,
            non_equality_tags=expr.non_equality_tags)


class TransferToNumpyMapper(CopyMapper):
    """A mapper to transfer arrays contained in :class:`~pytato.array.DataWrapper`
    instances to be :class:`numpy.ndarray` instances, using
    :meth:`~arraycontext.ArrayContext.to_numpy`.
    """
    def __init__(self, actx: ArrayContext) -> None:
        super().__init__()
        self.actx: ArrayContext = actx

    @override
    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        import numpy as np

        import arraycontext.impl.pyopencl.taggable_cl_array as tga
        if not isinstance(expr.data, tga.TaggableCLArray):
            raise ValueError("TransferToNumpyMapper: tried to transfer data that "
                             "is already on the host")

        np_data = self.actx.to_numpy(expr.data)
        assert isinstance(np_data, np.ndarray)

        return DataWrapper(
            data=np_data,
            shape=expr.shape,
            axes=expr.axes,
            tags=expr.tags,
            non_equality_tags=expr.non_equality_tags)


def transfer_from_numpy(expr: ArrayOrNamesTc, actx: ArrayContext) -> ArrayOrNamesTc:
    """Transfer arrays contained in :class:`~pytato.array.DataWrapper`
    instances to be device arrays, using
    :meth:`~arraycontext.ArrayContext.from_numpy`.
    """
    return TransferFromNumpyMapper(actx)(expr)


def transfer_to_numpy(expr: ArrayOrNamesTc, actx: ArrayContext) -> ArrayOrNamesTc:
    """Transfer arrays contained in :class:`~pytato.array.DataWrapper`
    instances to be :class:`numpy.ndarray` instances, using
    :meth:`~arraycontext.ArrayContext.to_numpy`.
    """
    return TransferToNumpyMapper(actx)(expr)

# }}}


# {{{ Profiling

def tabulate_profiling_data(actx: PytatoPyOpenCLArrayContext) -> pytools.Table:
    """Return a :class:`pytools.Table` with the profiling results."""
    actx._wait_and_transfer_profile_events()

    tbl = pytools.Table()

    # Table header
    tbl.add_row(("Kernel", "# Calls", "Time_sum [ns]", "Time_avg [ns]"))

    # Precision of results
    g = ".5g"

    total_calls = 0
    total_time = 0.0

    for kernel_name, times in actx._profile_results.items():
        num_calls = len(times)
        total_calls += num_calls

        t_sum = sum(times)
        t_avg = t_sum / num_calls
        total_time += t_sum

        tbl.add_row((kernel_name, num_calls, f"{t_sum:{g}}", f"{t_avg:{g}}"))

    tbl.add_row(("", "", "", ""))
    tbl.add_row(("Total", total_calls, f"{total_time:{g}}", "--"))

    actx._reset_profiling_data()

    return tbl

# }}}


# {{{ compile/outline helpers

def _ary_container_key_stringifier(keys: tuple[SerializationKey, ...]) -> str:
    """
    Helper for :meth:`BaseLazilyCompilingFunctionCaller.__call__`. Stringifies an
    array-container's component's key. Goals of this routine:

    * No two different keys should have the same stringification
    * Stringified key must a valid identifier according to :meth:`str.isidentifier`
    * (informal) Shorter identifiers are preferred
    """
    def _rec_str(key: object) -> str:
        if isinstance(key, str | int):
            return str(key)
        elif isinstance(key, tuple):
            # t in '_actx_t': stands for tuple
            return "_actx_t" + "_".join(_rec_str(k) for k in key) + "_actx_endt"  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        else:
            raise NotImplementedError("Key-stringication unimplemented for "
                                      f"'{type(key).__name__}'.")

    return "_".join(_rec_str(key) for key in keys)

# }}}

# vim: foldmethod=marker
