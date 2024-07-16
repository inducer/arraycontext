"""
.. currentmodule:: arraycontext

A :mod:`pytato`-based array context defers the evaluation of an array until its
frozen. The execution contexts for the evaluations are specific to an
:class:`~arraycontext.ArrayContext` type. For ex.
:class:`~arraycontext.ParamStudyPytatoPyOpenCLArrayContext`
uses :mod:`pyopencl` to JIT-compile and execute the array expressions.

Following :mod:`pytato`-based array context are provided:

.. autoclass:: ParamStudyPytatoPyOpenCLArrayContext

The compiled function is stored as.
.. autoclass:: ParamStudyLazyPyOpenCLFunctionCaller


Compiling a Python callable (Internal) for multiple distinct instances of
execution.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: arraycontext.parameter_study
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
    List,
)

import numpy as np
import pytato as pt

from pytools import memoize_method
from pytools.tag import Tag, ToTagSetConvertible, normalize_tags, UniqueTag

from dataclasses import dataclass

from arraycontext.container.traversal import (rec_map_array_container,
                                              with_array_context, rec_keyed_map_array_container)

from arraycontext.container import ArrayContainer, is_array_container_type

from arraycontext.context import ArrayT, ArrayContext
from arraycontext.metadata import NameHint
from arraycontext import PytatoPyOpenCLArrayContext
from arraycontext.impl.pytato.compile import (LazilyPyOpenCLCompilingFunctionCaller,
                                             _get_arg_id_to_arg_and_arg_id_to_descr,
                                                      _to_input_for_compiled,
                                              _ary_container_key_stringifier)

from arraycontext.parameter_study.transform import ExpansionMapper, ParameterStudyAxisTag

# from arraycontext.parameter_study.transform import ExpansionMapper

if TYPE_CHECKING:
    import pyopencl as cl
    import pytato

if getattr(sys, "_BUILDING_SPHINX_DOCS", False):
    import pyopencl as cl

import logging


logger = logging.getLogger(__name__)


def pack_for_parameter_study(actx: ArrayContext, study_name_tag: ParameterStudyAxisTag,
                             newshape: Tuple[int, ...],
                             *args: ArrayT) -> ArrayT:
    """
        Args is a list of variable names and the realized input data that needs
        to be packed for a parameter study or uncertainty quantification.

        Args needs to be in the format
            [v0, v1, v2, ..., vN] where N is the total number of instances you want to
            try. Note these may be across multiple parameter studies on the same inputs.
    """

    assert len(args) > 0
    assert len(args) == np.prod(newshape)

    orig_shape = args[0].shape
    out = actx.np.stack(args, axis=args[0].ndim)
    outshape = tuple(list(orig_shape) + [newshape] )

    #if len(newshape) > 1:
    #    # Reshape the object
    #    out = out.reshape(outshape)
    
    for i in range(len(orig_shape), len(outshape)):
        out = out.with_tagged_axis(i, [study_name_tag(i - len(orig_shape), newshape[i-len(orig_shape)])])
    return out


def unpack_parameter_study(data: ArrayT,
                           study_name_tag: ParameterStudyAxisTag) -> Dict[int,
                                                                          List[ArrayT]]:
    """
        Split the data array along the axes which vary according to a ParameterStudyAxisTag
        whose name tag is an instance study_name_tag.

        output[i] corresponds to the values associated with the ith parameter study that
        uses the variable name :arg: `study_name_tag`.
    """

    ndim: int = len(data.axes)
    out: Dict[int, List[ArrayT]] = {}

    study_count = 0
    for i in range(ndim):
        axis_tags = data.axes[i].tags_of_type(study_name_tag)
        if axis_tags:
            # Now we need to split this data.
            breakpoint()
            for j in range(data.shape[i]):
                tmp: List[slice] = [slice(None)] * ndim
                tmp[i] = j
                the_slice: Tuple[slice] = tuple(tmp)
                # Needs to be a tuple of slices not list of slices.
                if study_count in out.keys():
                    out[study_count].append(data[the_slice])
                else:
                    out[study_count] = [data[the_slice]]
            if study_count in out.keys():
                study_count += 1
                # yield data[the_slice]

    return out
