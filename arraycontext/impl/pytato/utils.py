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


from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Set, Tuple

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
from pytato.transform import CopyMapper
from pytools import UniqueNameGenerator, memoize_method

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
        self.bound_arguments: Dict[str, Any] = {}
        self.vng = UniqueNameGenerator()
        self.seen_inputs: Set[str] = set()

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
                    shape=tuple(self.rec(s) if isinstance(s, Array) else s
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
        ) -> Tuple[AbstractResultWithNamedArrays, Mapping[str, Any]]:
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
    return normalized_expr, normalize_mapper.bound_arguments


def get_pt_axes_from_cl_axes(axes: Tuple[ClAxis, ...]) -> Tuple[PtAxis, ...]:
    return tuple(PtAxis(axis.tags) for axis in axes)


def get_cl_axes_from_pt_axes(axes: Tuple[PtAxis, ...]) -> Tuple[ClAxis, ...]:
    return tuple(ClAxis(axis.tags) for axis in axes)


# {{{ arg-size-limiting loopy target

class ArgSizeLimitingPytatoLoopyPyOpenCLTarget(LoopyPyOpenCLTarget):
    def __init__(self, limit_arg_size_nbytes: int) -> None:
        super().__init__()
        self.limit_arg_size_nbytes = limit_arg_size_nbytes

    @memoize_method
    def get_loopy_target(self) -> Optional["lp.PyOpenCLTarget"]:
        from loopy import PyOpenCLTarget
        return PyOpenCLTarget(limit_arg_size_nbytes=self.limit_arg_size_nbytes)

# }}}


# {{{ compile/outline helpers

def _ary_container_key_stringifier(keys: Tuple[Any, ...]) -> str:
    """
    Helper for :meth:`BaseLazilyCompilingFunctionCaller.__call__`. Stringifies an
    array-container's component's key. Goals of this routine:

    * No two different keys should have the same stringification
    * Stringified key must a valid identifier according to :meth:`str.isidentifier`
    * (informal) Shorter identifiers are preferred
    """
    def _rec_str(key: Any) -> str:
        if isinstance(key, (str, int)):
            return str(key)
        elif isinstance(key, tuple):
            # t in '_actx_t': stands for tuple
            return "_actx_t" + "_".join(_rec_str(k) for k in key) + "_actx_endt"
        else:
            raise NotImplementedError("Key-stringication unimplemented for "
                                      f"'{type(key).__name__}'.")

    return "_".join(_rec_str(key) for key in keys)

# }}}

# vim: foldmethod=marker
