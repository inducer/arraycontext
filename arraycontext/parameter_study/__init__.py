"""
.. currentmodule:: arraycontext

A :mod:`pytato`-based array context defers the evaluation of an array until its
frozen. The execution contexts for the evaluations are specific to an
:class:`~arraycontext.ArrayContext` type. For ex.
:class:`~arraycontext.PytatoPyOpenCLArrayContext` uses :mod:`pyopencl` to
JIT-compile and execute the array expressions.

Following :mod:`pytato`-based array context are provided:

.. autoclass:: ParamStudyPytatoPyOpenCLArrayContext
.. autoclass:: ParamStudyLazyPyOpenCLFunctionCaller


Compiling a Python callable (Internal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: arraycontext.impl.pytato.compile
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

import abc
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Optional,
    Tuple,
    Type,
    Union,
    Sequence,
)

import numpy as np

from pytools import memoize_method
from pytools.tag import Tag, ToTagSetConvertible, normalize_tags, UniqueTag

from arraycontext.container.traversal import rec_map_array_container, with_array_context

from arraycontext.context import ArrayT, ArrayContext
from arraycontext.metadata import NameHint
from arraycontext.impl.pytato import PytatoPyOpenCLArrayContext
from arraycontext.impl.pytato.fake_numpy import PytatoFakeNumpyNamespace
from arraycontext.impl.pytato.compile import LazilyPyOpenCLCompilingFunctionCaller


from dataclasses import dataclass

if TYPE_CHECKING:
    import pyopencl as cl
    import pytato

if getattr(sys, "_BUILDING_SPHINX_DOCS", False):
    import pyopencl as cl

import logging


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ParameterStudyAxisTag(UniqueTag):
    """
        A tag for acting on axes of arrays.
    """
    user_variable_name: str
    axis_num: int
    axis_size: int

# {{{ ParamStudyPytatoPyOpenCLArrayContext


class ParamStudyPytatoPyOpenCLArrayContext(PytatoPyOpenCLArrayContext):
    """
    A derived class for PytatoPyOpenCLArrayContext updated for the
    purpose of enabling parameter studies and uncertainty quantification.

    .. automethod:: __init__

    .. automethod:: transform_dag

    .. automethod:: compile
    """

    def transform_dag(self, ary):
        # This going to be called before the compiler or freeze.
        out = super().transform_dag(ary)
        return out


# }}}


class ParamStudyLazyPyOpenCLFunctionCaller(LazilyPyOpenCLCompilingFunctionCaller):
    """
    Record a side-effect-free callable :attr:`f` which is initially designed for
    to be called multiple times with different data. This class will update the
    signature to allow :attr:`f` to be called once with the data for multiple
    instances.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Returns the result of :attr:`~BaseLazilyCompilingFunctionCaller.f`'s
        function application on *args*.

        Before applying :attr:`~BaseLazilyCompilingFunctionCaller.f`, it is compiled
        to a :mod:`pytato` DAG that would apply
        :attr:`~BaseLazilyCompilingFunctionCaller.f` with *args* in a lazy-sense.
        The intermediary pytato DAG for *args* is memoized in *self*.
        """
        arg_id_to_arg, arg_id_to_descr = _get_arg_id_to_arg_and_arg_id_to_descr(
            args, kwargs)

        try:
            compiled_f = self.program_cache[arg_id_to_descr]
        except KeyError:
            pass
        else:
            return compiled_f(arg_id_to_arg)

        dict_of_named_arrays = {}
        output_id_to_name_in_program = {}
        input_id_to_name_in_program = {
            arg_id: f"_actx_in_{_ary_container_key_stringifier(arg_id)}"
            for arg_id in arg_id_to_arg}

        output_template = self.f(
                *[_get_f_placeholder_args(arg, iarg,
                                          input_id_to_name_in_program, self.actx)
                    for iarg, arg in enumerate(args)],
                **{kw: _get_f_placeholder_args(arg, kw,
                                               input_id_to_name_in_program,
                                               self.actx)
                    for kw, arg in kwargs.items()})

        self.actx._compile_trace_callback(self.f, "post_trace", output_template)

        if (not (is_array_container_type(output_template.__class__)
                 or isinstance(output_template, pt.Array))):
            # TODO: We could possibly just short-circuit this interface if the
            # returned type is a scalar. Not sure if it's worth it though.
            raise NotImplementedError(
                f"Function '{self.f.__name__}' to be compiled "
                "did not return an array container or pt.Array,"
                f" but an instance of '{output_template.__class__}' instead.")

        def _as_dict_of_named_arrays(keys, ary):
            name = "_pt_out_" + _ary_container_key_stringifier(keys)
            output_id_to_name_in_program[keys] = name
            dict_of_named_arrays[name] = ary
            return ary

        rec_keyed_map_array_container(_as_dict_of_named_arrays,
                                      output_template)

        compiled_func = self._dag_to_compiled_func(
                pt.make_dict_of_named_arrays(dict_of_named_arrays),
                input_id_to_name_in_program=input_id_to_name_in_program,
                output_id_to_name_in_program=output_id_to_name_in_program,
                output_template=output_template)

        self.program_cache[arg_id_to_descr] = compiled_func
        return compiled_func(arg_id_to_arg)


def _to_input_for_compiled(ary: ArrayT, actx: PytatoPyOpenCLArrayContext):
    """
    Preprocess *ary* before turning it into a :class:`pytato.array.Placeholder`
    in :meth:`LazilyCompilingFunctionCaller.__call__`.

    Preprocessing here refers to:

    - Metadata Inference that is supplied via *actx*\'s
      :meth:`PytatoPyOpenCLArrayContext.transform_dag`.
    """
    import pyopencl.array as cla

    from arraycontext.impl.pyopencl.taggable_cl_array import (
        TaggableCLArray,
        to_tagged_cl_array,
    )
    if isinstance(ary, pt.Array):
        dag = pt.make_dict_of_named_arrays({"_actx_out": ary})
        # Transform the DAG to give metadata inference a chance to do its job
        return actx.transform_dag(dag)["_actx_out"].expr
    elif isinstance(ary, TaggableCLArray):
        return ary
    elif isinstance(ary, cla.Array):
        from warnings import warn
        warn("Passing pyopencl.array.Array to a compiled callable"
             " is deprecated and will stop working in 2023."
             " Use `to_tagged_cl_array` to convert the array to"
             " TaggableCLArray", DeprecationWarning, stacklevel=2)

        return to_tagged_cl_array(ary,
                                  axes=None,
                                  tags=frozenset())
    else:
        raise NotImplementedError(type(ary))


def _get_f_placeholder_args(arg, kw, arg_id_to_name, actx):
    """
    Helper for :class:`BaseLazilyCompilingFunctionCaller.__call__`. Returns the
    placeholder version of an argument to
    :attr:`BaseLazilyCompilingFunctionCaller.f`.
    """
    if np.isscalar(arg):
        name = arg_id_to_name[(kw,)]
        return pt.make_placeholder(name, (), np.dtype(type(arg)))
    elif isinstance(arg, pt.Array):
        name = arg_id_to_name[(kw,)]
        # Transform the DAG to give metadata inference a chance to do its job
        arg = _to_input_for_compiled(arg, actx)
        return pt.make_placeholder(name, arg.shape, arg.dtype,
                                   axes=arg.axes,
                                   tags=arg.tags)
    elif is_array_container_type(arg.__class__):
        def _rec_to_placeholder(keys, ary):
            name = arg_id_to_name[(kw, *keys)]
            # Transform the DAG to give metadata inference a chance to do its job
            ary = _to_input_for_compiled(ary, actx)
            return pt.make_placeholder(name,
                                       ary.shape,
                                       ary.dtype,
                                       axes=ary.axes,
                                       tags=ary.tags)

        return rec_keyed_map_array_container(_rec_to_placeholder, arg)
    else:
        raise NotImplementedError(type(arg))


def pack_for_parameter_study(actx: ArrayContext, yourvarname: str,
                             newshape: Tuple[int, ...],
                             *args: ArrayT) -> ArrayT:
    """
        Args is a list of variable names and the realized input data that needs
        to be packed for a parameter study or uncertainty quantification.

        Args needs to be in the format
            ["v", v0, v1, v2, ..., vN, "w", w0, w1, w2, ..., wM, \dots]

            where "v" and "w" would be the variable names in your program.
            If you want to include a constant just pass the var name and then
            the value in the next argument.

        Returns a dictionary of {var name: stacked array}
    """

    assert len(args) > 0
    assert len(args) == np.prod(newshape)

    out = {}
    orig_shape = args[0].shape
    out = actx.np.stack(args)
    outshape = tuple([newshape] + [val for val in orig_shape])

    if len(newshape) > 1:
        # Reshape the object
        out = out.reshape(outshape)
    for i in range(len(newshape)):
        out = out.with_tagged_axis(i, [ParameterStudyAxisTag(yourvarname, i, newshape[i])])
    return out


def unpack_parameter_study(data: ArrayT, varname: str) -> Sequence[ArrayT]:
    """
        Split the data array along the axes which vary according to a ParameterStudyAxisTag
        whose variable name is varname.
    """

    ndim = len(data.axes)
    out = {}

    for i in range(ndim):
        axis_tags = data.axes[i].tags_of_type(ParameterStudyAxisTag)
        if axis_tags:
            # Now we need to split this data.
            breakpoint()
            for j in range(data.shape[i]):
                the_slice = [slice(None)] * ndim
                the_slice[i] = j
                the_slice = tuple(the_slice)
                out[tuple([i,j])] = data[the_slice]
                #yield data[the_slice]

    return out
