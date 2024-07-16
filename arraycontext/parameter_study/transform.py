"""
.. currentmodule:: arraycontext

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
    Mapping,
)

import numpy as np
import pytato as pt
import loopy as lp
from immutabledict import immutabledict


from pytato.scalar_expr import IdentityMapper
import pymbolic.primitives as prim


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


from pytato.transform import CopyMapper

from pytato.array import (
        Array, IndexLambda, Placeholder, Stack, Roll, AxisPermutation,
        DataWrapper, SizeParam, DictOfNamedArrays, AbstractResultWithNamedArrays,
        Reshape, Concatenate, NamedArray, IndexRemappingBase, Einsum,
        InputArgumentBase, AdvancedIndexInNoncontiguousAxes, IndexBase, DataInterface,
        Axis)

from pytato.utils import broadcast_binary_op

@dataclass(frozen=True)
class ParameterStudyAxisTag(UniqueTag):
    """
        A tag for acting on axes of arrays.
        To enable multiple parameter studies on the same variable name
        specify a different axis number and potentially a different size.

        Currently does not allow multiple variables of different names to be in
        the same parameter study.
    """
    #user_param_study_tag: Tag 
    axis_num: int
    axis_size: int

class ExpansionMapper(CopyMapper):

    #def __init__(self, dependency_map: Dict[Array,Tag]):
        #    super().__init__()
    #    self.depends = dependency_map
    def __init__(self, actual_input_shapes: Mapping[str, Tuple[int,...]],
                 actual_input_axes: Mapping[str, FrozenSet[Axis]]):
        super().__init__()
        self.actual_input_shapes = actual_input_shapes
        self.actual_input_axes = actual_input_axes


    def does_single_predecessor_require_rewrite_of_this_operation(self, curr_expr: Array,
                                                                  new_expr: Array) -> Tuple[Optional[Tuple[int]],
                                                                                            Optional[Tuple[Axis]]]:
        shape_to_prepend: Tuple[int] = tuple([])
        new_axes: Tuple[Axis] = tuple([])
        if curr_expr.shape == new_expr.shape:
            return shape_to_prepend, new_axes
        
        # Now we may need to change.
        changed = False
        for i in range(len(new_expr.axes)):
            axis_tags = list(new_expr.axes[i].tags)
            for j, tag in enumerate(axis_tags):
                # Should be relatively few tags on each axis $O(1)$.
                if isinstance(tag, ParameterStudyAxisTag):
                    new_axes = new_axes + (new_expr.axes[i],)
                    shape_to_prepend = shape_to_prepend + (new_expr.shape[i],)
        return shape_to_prepend, new_axes


    def map_stack(self, expr: Stack) -> Array:
        # TODO: Fix
        return super().map_stack(expr)

    def map_concatenate(self, expr: Concatenate) -> Array:
        return super().map_concatenate(expr)

    def map_roll(self, expr: Roll) -> Array:
        new_array = self.rec(expr.array)
        prepend_shape, new_axes =self.does_single_predecessor_require_rewrite_of_this_operation(expr.array,
                                                                                                new_array)
        return Roll(array=new_array,
                    shift=expr.shift,
                    axis=expr.axis + len(new_axes),
                    axes=new_axes + expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> Array:
        new_array = self.rec(expr.array)
        prepend_shape, new_axes = self.does_single_predecessor_require_rewrite_of_this_operation(expr.array,
                                                                                                 new_array)
        axis_permute = tuple([expr.axis_permutation[i] + len(prepend_shape) for i
                              in range(len(expr.axis_permutation))])
        # Include the axes we are adding to the system.
        axis_permute = tuple([i for i in range(len(prepend_shape))]) + axis_permute


        return AxisPermutation(array=new_array,
                               axis_permutation=axis_permute,
                               axes=new_axes + expr.axes,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def _map_index_base(self, expr: IndexBase) -> Array:
        new_array = self.rec(expr.array)
        prepend_shape, new_axes = self.does_single_predecessor_require_rewrite_of_this_operation(expr.array,
                                                                                                 new_array)
        return type(expr)(new_array,
                          indices=self.rec_idx_or_size_tuple(expr.indices),
                          # May need to modify indices
                          axes=new_axes + expr.axes,
                          tags=expr.tags,
                          non_equality_tags = expr.non_equality_tags)

    def map_reshape(self, expr: Reshape) -> Array:
        new_array = self.rec(expr.array)
        prepend_shape, new_axes = self.does_single_predecessor_require_rewrite_of_this_operation(expr.array,
                                                                                                 new_array) 
        return Reshape(new_array,
                       newshape = self.rec_idx_or_size_tuple(prepend_shape + expr.newshape),
                       order=expr.order,
                       axes=new_axes + expr.axes,
                       tags=expr.tags,
                       non_equality_tags=expr.non_equality_tags)

    def map_placeholder(self, expr: Placeholder) -> Array:
        # This is where we could introduce extra axes.
        correct_shape = expr.shape
        correct_axes = expr.axes
        if expr.name in self.actual_input_shapes.keys():
            # We may need to update the size.
            if expr.shape != self.actual_input_shapes[expr.name]:
                correct_shape = self.actual_input_shapes[expr.name]
                correct_axes = self.actual_input_axes[expr.name]
        return Placeholder(name=expr.name,
                           shape=self.rec_idx_or_size_tuple(correct_shape),
                           dtype=expr.dtype,
                           axes=correct_axes,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        # Update bindings first.
        new_bindings: Mapping[str, Array] = { name: self.rec(bnd) 
                                             for name, bnd in sorted(expr.bindings.items())}

        # Determine the new parameter studies that are being conducted.
        from pytools import unique
        from pytools.obj_array import flat_obj_array

        all_axis_tags: Set[Tag] = set()
        studies_by_variable: Mapping[str, Mapping[Tag, bool]] = {}
        for name, bnd in sorted(new_bindings.items()):
            axis_tags_for_bnd: Set[Tag] = set()
            studies_by_variable[name] = {}
            for i in range(len(bnd.axes)):
                axis_tags_for_bnd = axis_tags_for_bnd.union(bnd.axes[i].tags_of_type(ParameterStudyAxisTag))
            for tag in axis_tags_for_bnd:
                studies_by_variable[name][tag] = 1
            all_axis_tags = all_axis_tags.union(axis_tags_for_bnd)

        # Freeze the set now.
        all_axis_tags = frozenset(all_axis_tags)

        active_studies: Sequence[ParameterStudyAxisTag] = list(unique(all_axis_tags))
        axes: Optional[Tuple[Axis]] = tuple([])
        study_to_axis_number: Mapping[ParameterStudyAxisTag, int] = {}

        count = 0
        new_shape = expr.shape
        new_axes  = expr.axes

        for study in active_studies:
            if isinstance(study, ParameterStudyAxisTag):
                # Just defensive programming
                # The active studies are added to the end of the bindings.
                study_to_axis_number[study] = len(new_shape)
                new_shape = new_shape + (study.axis_size,)
                new_axes = new_axes + (Axis(tags=frozenset((study,))),)
                #  This assumes that the axis only has 1 tag,
                #  because there should be no dependence across instances.

        # Now we need to update the expressions.
        scalar_expr = ParamAxisExpander()(expr.expr, studies_by_variable, study_to_axis_number)

        return IndexLambda(expr=scalar_expr,
                           bindings=type(expr.bindings)(new_bindings),
                           shape=new_shape,
                           var_to_reduction_descr=expr.var_to_reduction_descr,
                           dtype=expr.dtype,
                           axes=new_axes,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)


class ParamAxisExpander(IdentityMapper):

    def map_subscript(self, expr: prim.Subscript, studies_by_variable: Mapping[str, Mapping[ParameterStudyAxisTag, bool]],
                      study_to_axis_number: Mapping[ParameterStudyAxisTag, int]):
        # We know that we are not changing the variable that we are indexing into.
        # This is stored in the aggregate member of the class Subscript.

        # We only need to modify the indexing which is stored in the index member.
        name = expr.aggregate.name
        if name in studies_by_variable.keys():
            #  These are the single instance information.
            index = self.rec(expr.index, studies_by_variable,
                             study_to_axis_number)
            
            new_vars: Tuple[prim.Variable] = tuple([])

            for key, val in sorted(study_to_axis_number.items(), key=lambda item: item[1]):
                if key in studies_by_variable[name]:
                    new_vars = new_vars + (prim.Variable(f"_{study_to_axis_number[key]}"),)

            if isinstance(index, tuple):
                index = index + new_vars
            else:
                index = tuple(index) + new_vars
            return type(expr)(aggregate=expr.aggregate, index=index)
        return expr


# {{{ ParamStudyPytatoPyOpenCLArrayContext


class ParamStudyPytatoPyOpenCLArrayContext(PytatoPyOpenCLArrayContext):
    """
    A derived class for PytatoPyOpenCLArrayContext updated for the
    purpose of enabling parameter studies and uncertainty quantification.

    .. automethod:: __init__

    .. automethod:: transform_dag

    .. automethod:: compile
    """

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        return ParamStudyLazyPyOpenCLFunctionCaller(self, f)


    def transform_loopy_program(self, t_unit: lp.TranslationUnit) -> lp.TranslationUnit:
        # Update in a subclass if you want.
        return t_unit

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
        Returns the result of :attr:`~ParamStudyLazyPyOpenCLFunctionCaller.f`'s
        function application on *args*.

        Before applying :attr:`~ParamStudyLazyPyOpenCLFunctionCaller.f`, it is compiled
        to a :mod:`pytato` DAG that would apply
        :attr:`~ParamStudyLazyPyOpenCLFunctionCaller.f` with *args* in a lazy-sense.
        The intermediary pytato DAG for *args* is memoized in *self*.
        """
        arg_id_to_arg, arg_id_to_descr = _get_arg_id_to_arg_and_arg_id_to_descr(
            args, kwargs)

        try:
            compiled_f = self.program_cache[arg_id_to_descr]
        except KeyError:
            pass
        else:
            # On a cache hit we do not need to modify anything.
            return compiled_f(arg_id_to_arg)

        dict_of_named_arrays = {}
        output_id_to_name_in_program = {}
        input_id_to_name_in_program = {
            arg_id: f"_actx_in_{_ary_container_key_stringifier(arg_id)}"
            for arg_id in arg_id_to_arg}

        output_template = self.f(
                *[_get_f_placeholder_args_for_param_study(arg, iarg,
                                          input_id_to_name_in_program, self.actx)
                    for iarg, arg in enumerate(args)],
                **{kw: _get_f_placeholder_args_for_param_study(arg, kw,
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

        input_shapes = {input_id_to_name_in_program[i]: arg_id_to_descr[i].shape for i in arg_id_to_descr.keys()}
        input_axes = {input_id_to_name_in_program[i]: arg_id_to_arg[i].axes for i in arg_id_to_descr.keys()}
        myMapper = ExpansionMapper(input_shapes, input_axes) # Get the dependencies
        breakpoint()

        dict_of_named_arrays = pt.make_dict_of_named_arrays(dict_of_named_arrays)

        breakpoint()
        dict_of_named_arrays = myMapper(dict_of_named_arrays) # Update the arrays.

        # Use the normal compiler now.
        
        compiled_func = self._dag_to_compiled_func(dict_of_named_arrays,
                input_id_to_name_in_program=input_id_to_name_in_program,
                output_id_to_name_in_program=output_id_to_name_in_program,
                output_template=output_template)

        breakpoint()
        self.program_cache[arg_id_to_descr] = compiled_func
        return compiled_func(arg_id_to_arg)


def _cut_if_in_param_study(name, arg) -> Array:
    """
        Helper to split a place holder into the base instance shape
        if it is tagged with a `ParameterStudyAxisTag`
        to ensure the survival of the information those tags will be converted
        to temporary Array Tags of the same type. The placeholder will not
        have the axes marked with a `ParameterStudyAxisTag` tag. 
    """
    ndim: int = len(arg.shape)
    newshape = []
    update_axes = []
    for i in range(ndim):
        axis_tags = arg.axes[i].tags_of_type(ParameterStudyAxisTag)
        if not axis_tags:
            update_axes.append(arg.axes[i])
            newshape.append(arg.shape[i])
    
    update_axes = tuple(update_axes)
    update_tags: FrozenSet[Tag] = arg.tags
    return pt.make_placeholder(name, newshape, arg.dtype, axes=update_axes,
                               tags=update_tags)

def _get_f_placeholder_args_for_param_study(arg, kw, arg_id_to_name, actx):
    """
    Helper for :class:`BaseLazilyCompilingFunctionCaller.__call__`. 
    Returns the placeholder version of an argument to
    :attr:`ParamStudyLazyPyOpenCLFunctionCaller.f`.
    
    Note this will modify the shape of the placeholder to
    remove any parameter study axes until the trace
    can be completed.

    They will be added back after the trace is complete.
    """
    if np.isscalar(arg):
        name = arg_id_to_name[(kw,)]
        return pt.make_placeholder(name, (), np.dtype(type(arg)))
    elif isinstance(arg, pt.Array):
        name = arg_id_to_name[(kw,)]
        # Transform the DAG to give metadata inference a chance to do its job
        arg = _to_input_for_compiled(arg, actx)
        return _cut_if_in_param_study(name, arg)
    elif is_array_container_type(arg.__class__):
        def _rec_to_placeholder(keys, ary):
            name = arg_id_to_name[(kw, *keys)]
            # Transform the DAG to give metadata inference a chance to do its job
            ary = _to_input_for_compiled(ary, actx)
            return _cut_if_in_param_study(name, ary)

        return rec_keyed_map_array_container(_rec_to_placeholder, arg)
    else:
        raise NotImplementedError(type(arg))
