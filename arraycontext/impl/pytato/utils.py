__doc__ = """
.. autofunction:: transfer_from_numpy
.. autofunction:: transfer_to_numpy
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


from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from pytato.array import (
    AbstractResultWithNamedArrays,
    Array,
    Axis as PtAxis,
    DataWrapper,
    DictOfNamedArrays,
    Placeholder,
    SizeParam,
    make_placeholder,
)
from pytato.target.loopy import LoopyPyOpenCLTarget
from pytato.transform import ArrayOrNames, CopyMapper
from pytools import UniqueNameGenerator, memoize_method

from arraycontext import ArrayContext
from arraycontext.impl.pyopencl.taggable_cl_array import Axis as ClAxis


if TYPE_CHECKING:
    import loopy as lp


class _DatawrapperToBoundPlaceholderMapper(CopyMapper):
    """
    Helper mapper for :func:`normalize_pt_expr`. Every
    :class:`pytato.DataWrapper` is replaced with a deterministic copy of
    :class:`Placeholder`.
    """
    def __init__(self) -> None:
        super().__init__()
        self.bound_arguments: dict[str, Any] = {}
        self.vng = UniqueNameGenerator()
        self.seen_inputs: set[str] = set()

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
                    shape=tuple(cast(Array, self.rec(s)) if isinstance(s, Array) else s
                                for s in expr.shape),
                    dtype=expr.dtype,
                    axes=expr.axes,
                    tags=expr.tags)

    def map_size_param(self, expr: SizeParam) -> Array:
        raise NotImplementedError

    def map_placeholder(self, expr: Placeholder) -> Array:
        raise ValueError("Placeholders cannot appear in"
                         " DatawrapperToBoundPlaceholderMapper.")


def _normalize_pt_expr(
        expr: DictOfNamedArrays
        ) -> tuple[Array | AbstractResultWithNamedArrays, Mapping[str, Any]]:
    """
    Returns ``(normalized_expr, bound_arguments)``.  *normalized_expr* is a
    normalized form of *expr*, with all instances of
    :class:`pytato.DataWrapper` replaced with instances of :class:`Placeholder`
    named in a deterministic manner. The data corresponding to the placeholders
    in *normalized_expr* is recorded in the mapping *bound_arguments*.
    Deterministic naming of placeholders permits more effective caching of
    equivalent graphs.
    """
    normalize_mapper = _DatawrapperToBoundPlaceholderMapper()
    normalized_expr = normalize_mapper(expr)
    assert isinstance(normalized_expr, AbstractResultWithNamedArrays)
    return normalized_expr, normalize_mapper.bound_arguments


def get_pt_axes_from_cl_axes(axes: tuple[ClAxis, ...]) -> tuple[PtAxis, ...]:
    return tuple(PtAxis(axis.tags) for axis in axes)


def get_cl_axes_from_pt_axes(axes: tuple[PtAxis, ...]) -> tuple[ClAxis, ...]:
    return tuple(ClAxis(axis.tags) for axis in axes)


# {{{ arg-size-limiting loopy target

class ArgSizeLimitingPytatoLoopyPyOpenCLTarget(LoopyPyOpenCLTarget):
    def __init__(self, limit_arg_size_nbytes: int) -> None:
        super().__init__()
        self.limit_arg_size_nbytes = limit_arg_size_nbytes

    @memoize_method
    def get_loopy_target(self) -> "lp.PyOpenCLTarget":
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
        self.actx = actx

    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        import numpy as np

        if not isinstance(expr.data, np.ndarray):
            raise ValueError("TransferFromNumpyMapper: tried to transfer data that "
                             "is already on the device")

        # Ideally, this code should just do
        # return self.actx.from_numpy(expr.data).tagged(expr.tags),
        # but there seems to be no way to transfer the non_equality_tags in that case.
        new_dw = self.actx.from_numpy(expr.data)
        assert isinstance(new_dw, DataWrapper)

        # https://github.com/pylint-dev/pylint/issues/3893
        # pylint: disable=unexpected-keyword-arg
        return DataWrapper(
            data=new_dw.data,
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
        self.actx = actx

    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        import numpy as np

        import arraycontext.impl.pyopencl.taggable_cl_array as tga
        if not isinstance(expr.data, tga.TaggableCLArray):
            raise ValueError("TransferToNumpyMapper: tried to transfer data that "
                             "is already on the host")

        np_data = self.actx.to_numpy(expr.data)
        assert isinstance(np_data, np.ndarray)

        # https://github.com/pylint-dev/pylint/issues/3893
        # pylint: disable=unexpected-keyword-arg
        return DataWrapper(
            data=np_data,
            shape=expr.shape,
            axes=expr.axes,
            tags=expr.tags,
            non_equality_tags=expr.non_equality_tags)


def transfer_from_numpy(expr: ArrayOrNames, actx: ArrayContext) -> ArrayOrNames:
    """Transfer arrays contained in :class:`~pytato.array.DataWrapper`
    instances to be device arrays, using
    :meth:`~arraycontext.ArrayContext.from_numpy`.
    """
    return TransferFromNumpyMapper(actx)(expr)


def transfer_to_numpy(expr: ArrayOrNames, actx: ArrayContext) -> ArrayOrNames:
    """Transfer arrays contained in :class:`~pytato.array.DataWrapper`
    instances to be :class:`numpy.ndarray` instances, using
    :meth:`~arraycontext.ArrayContext.to_numpy`.
    """
    return TransferToNumpyMapper(actx)(expr)

# }}}

# vim: foldmethod=marker
