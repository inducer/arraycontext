"""
.. currentmodule:: arraycontext
.. autofunction:: make_loopy_program
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

from typing import ClassVar, Mapping

import numpy as np

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION
from pytools import memoize_in

from arraycontext.container.traversal import multimapped_over_array_containers
from arraycontext.fake_numpy import BaseFakeNumpyNamespace


# {{{ loopy

_DEFAULT_LOOPY_OPTIONS = lp.Options(
        no_numpy=True,
        return_dict=True)


def make_loopy_program(domains, statements, kernel_data=None,
        name="mm_actx_kernel", tags=None):
    """Return a :class:`loopy.LoopKernel` suitable for use with
    :meth:`ArrayContext.call_loopy`.
    """
    if kernel_data is None:
        kernel_data = ["..."]

    return lp.make_kernel(
            domains,
            statements,
            kernel_data=kernel_data,
            options=_DEFAULT_LOOPY_OPTIONS,
            default_offset=lp.auto,
            name=name,
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            tags=tags)


def get_default_entrypoint(t_unit):
    try:
        # main and "kernel callables" branch
        return t_unit.default_entrypoint
    except AttributeError:
        try:
            return t_unit.root_kernel
        except AttributeError as err:
            raise TypeError("unable to find default entry point for loopy "
                    "translation unit") from err


def _get_scalar_func_loopy_program(actx, c_name, nargs, naxes):
    @memoize_in(actx, _get_scalar_func_loopy_program)
    def get(c_name, nargs, naxes):
        from pymbolic import var

        var_names = [f"i{i}" for i in range(naxes)]
        size_names = [f"n{i}" for i in range(naxes)]
        subscript = tuple(var(vname) for vname in var_names)
        from islpy import make_zero_and_vars
        v = make_zero_and_vars(var_names, params=size_names)
        domain = v[0].domain()
        for vname, sname in zip(var_names, size_names):
            domain = domain & v[0].le_set(v[vname]) & v[vname].lt_set(v[sname])

        domain_bset, = domain.get_basic_sets()

        import loopy as lp

        from .loopy import make_loopy_program
        from arraycontext.transform_metadata import ElementwiseMapKernelTag
        return make_loopy_program(
                [domain_bset],
                [
                    lp.Assignment(
                        var("out")[subscript],
                        var(c_name)(*[
                            var(f"inp{i}")[subscript] for i in range(nargs)]))
                    ],
                [
                    lp.GlobalArg("out",
                                 dtype=None, shape=lp.auto, offset=lp.auto)] + [
                        lp.GlobalArg(f"inp{i}",
                                     dtype=None, shape=lp.auto, offset=lp.auto)
                        for i in range(nargs)] + [...],
                name=f"actx_special_{c_name}",
                tags=(ElementwiseMapKernelTag(),))

    return get(c_name, nargs, naxes)


class LoopyBasedFakeNumpyNamespace(BaseFakeNumpyNamespace):
    _numpy_to_c_arc_functions: ClassVar[Mapping[str, str]] = {
            "arcsin": "asin",
            "arccos": "acos",
            "arctan": "atan",
            "arctan2": "atan2",

            "arcsinh": "asinh",
            "arccosh": "acosh",
            "arctanh": "atanh",
            }

    _c_to_numpy_arc_functions: ClassVar[Mapping[str, str]] = {c_name: numpy_name
            for numpy_name, c_name in _numpy_to_c_arc_functions.items()}

    def __getattr__(self, name):
        def loopy_implemented_elwise_func(*args):
            if all(np.isscalar(ary) for ary in args):
                return getattr(
                         np, self._c_to_numpy_arc_functions.get(name, name)
                         )(*args)
            actx = self._array_context
            prg = _get_scalar_func_loopy_program(actx,
                    c_name, nargs=len(args), naxes=len(args[0].shape))
            outputs = actx.call_loopy(prg,
                    **{f"inp{i}": arg for i, arg in enumerate(args)})
            return outputs["out"]

        if name in self._c_to_numpy_arc_functions:
            raise RuntimeError(f"'{name}' in ArrayContext.np has been removed. "
                    f"Use '{self._c_to_numpy_arc_functions[name]}' as in numpy. ")

        # normalize to C names anyway
        c_name = self._numpy_to_c_arc_functions.get(name, name)

        # limit which functions we try to hand off to loopy
        if (name in self._numpy_math_functions
                or name in self._c_to_numpy_arc_functions):
            return multimapped_over_array_containers(loopy_implemented_elwise_func)
        else:
            raise AttributeError(
                    f"'{type(self._array_context).__name__}.np' object "
                    f"has no attribute '{name}'")

# }}}


# vim: foldmethod=marker
