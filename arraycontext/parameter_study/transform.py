from __future__ import annotations


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

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Mapping,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pymbolic.primitives as prim
from immutabledict import immutabledict

import loopy as lp
import pytato as pt
from pytato.array import (
    Array,
    AxesT,
    Axis,
    AxisPermutation,
    Concatenate,
    Einsum,
    EinsumElementwiseAxis,
    IndexBase,
    IndexLambda,
    Placeholder,
    Reshape,
    Roll,
    ShapeType,
    Stack,
)
from pytato.scalar_expr import IdentityMapper
from pytato.transform import CopyMapper
from pytools.tag import Tag, UniqueTag

from arraycontext import PytatoPyOpenCLArrayContext
from arraycontext.container import is_array_container_type
from arraycontext.container.traversal import rec_keyed_map_array_container
from arraycontext.impl.pytato.compile import (
    LazilyPyOpenCLCompilingFunctionCaller,
    LeafArrayDescriptor,
    _ary_container_key_stringifier,
    _get_arg_id_to_arg_and_arg_id_to_descr,
    _to_input_for_compiled,
)


ArraysT = Tuple[Array, ...]


@dataclass(frozen=True)
class ParameterStudyAxisTag(UniqueTag):
    """
        A tag for acting on axes of arrays.
        To enable multiple parameter studies on the same variable name
        specify a different axis number and potentially a different size.

        Currently does not allow multiple variables of different names to be in
        the same parameter study.
    """
    # user_param_study_tag: Tag
    axis_num: int
    axis_size: int


StudiesT = Tuple[ParameterStudyAxisTag, ...]


class ExpansionMapper(CopyMapper):

    def __init__(self, actual_input_shapes: Mapping[str, ShapeType],
                 actual_input_axes: Mapping[str, FrozenSet[Axis]]):
        super().__init__()
        self.actual_input_shapes = actual_input_shapes
        self.actual_input_axes = actual_input_axes

    def single_predecessor_updates(self, curr_expr: Array,
                                          new_expr: Array) -> Tuple[ShapeType,
                                                                    AxesT]:
        # Initialize with something for the typing.
        shape_to_append: ShapeType = (-1,)
        new_axes: AxesT = (Axis(tags=frozenset()),)
        if curr_expr.shape == new_expr.shape:
            return shape_to_append, new_axes

        # Now we may need to change.
        for i in range(len(new_expr.axes)):
            axis_tags = list(new_expr.axes[i].tags)
            already_added = False
            for _j, tag in enumerate(axis_tags):
                # Should be relatively few tags on each axis $O(1)$.
                if isinstance(tag, ParameterStudyAxisTag):
                    new_axes = *new_axes, new_expr.axes[i],
                    shape_to_append = *shape_to_append, new_expr.shape[i],
                    if already_added:
                        raise ValueError("An individual axis may only be " +
                                "tagged with one ParameterStudyAxisTag.")
                    already_added = True

        # Remove initialized extraneous data
        return shape_to_append[1:], new_axes[1:]

    def map_roll(self, expr: Roll) -> Array:
        new_array = self.rec(expr.array)
        _, new_axes = self.single_predecessor_updates(expr.array, new_array)
        return Roll(array=new_array,
                    shift=expr.shift,
                    axis=expr.axis,
                    axes=expr.axes + new_axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> Array:
        new_array = self.rec(expr.array)
        postpend_shape, new_axes = self.single_predecessor_updates(expr.array, new_array)
        # Include the axes we are adding to the system.
        axis_permute = expr.axis_permutation + tuple([i + len(expr.axis_permutation)
                                             for i in range(len(postpend_shape))])

        return AxisPermutation(array=new_array,
                               axis_permutation=axis_permute,
                               axes=expr.axes + new_axes,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def _map_index_base(self, expr: IndexBase) -> Array:
        breakpoint()
        new_array = self.rec(expr.array)
        _, new_axes = self.single_predecessor_updates(expr.array, new_array)
        return type(expr)(new_array,
                          indices=self.rec_idx_or_size_tuple(expr.indices),
                          # May need to modify indices
                          axes=expr.axes + new_axes,
                          tags=expr.tags,
                          non_equality_tags=expr.non_equality_tags)

    def map_reshape(self, expr: Reshape) -> Array:
        new_array = self.rec(expr.array)
        postpend_shape, new_axes = self.single_predecessor_updates(expr.array, new_array)
        return Reshape(new_array,
                       newshape=self.rec_idx_or_size_tuple(expr.newshape + \
                                                           postpend_shape),
                       order=expr.order,
                       axes=expr.axes + new_axes,
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
                correct_axes = tuple(self.actual_input_axes[expr.name])
        return Placeholder(name=expr.name,
                           shape=self.rec_idx_or_size_tuple(correct_shape),
                           dtype=expr.dtype,
                           axes=correct_axes,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    # {{{ Operations with multiple predecessors.

    def _studies_from_multiple_pred(self,
                                    new_arrays: ArraysT) -> Tuple[AxesT,
                                                                  Set[ParameterStudyAxisTag],
                                                                  Dict[Array,
                                                                       StudiesT]]:

        new_axes_for_end: AxesT = ()
        cur_studies: Set[ParameterStudyAxisTag] = set()
        studies_by_array: Dict[Array, StudiesT] = {}

        for _ind, array in enumerate(new_arrays):
            for axis in array.axes:
                axis_tags = axis.tags_of_type(ParameterStudyAxisTag)
                if axis_tags:
                    axis_tags = list(axis_tags)
                    assert len(axis_tags) == 1
                    if array in studies_by_array.keys():
                        studies_by_array[array] = studies_by_array[array] + \
                                                (axis_tags[0],)
                    else:
                        studies_by_array[array] = (axis_tags[0],)

                    if axis_tags[0] not in cur_studies:
                        cur_studies.add(axis_tags[0])
                        new_axes_for_end = *new_axes_for_end, axis

        return new_axes_for_end, cur_studies, studies_by_array

    def map_stack(self, expr: Stack) -> Array:
        new_arrays, new_axes_for_end = self._mult_pred_same_shape(expr)

        return Stack(arrays=new_arrays,
                     axis=expr.axis,
                     axes=expr.axes + new_axes_for_end,
                     tags=expr.tags,
                     non_equality_tags=expr.non_equality_tags)

    def map_concatenate(self, expr: Concatenate) -> Array:
        new_arrays, new_axes_for_end = self._mult_pred_same_shape(expr)

        return Concatenate(arrays=new_arrays,
                           axis=expr.axis,
                           axes=expr.axes + new_axes_for_end,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def _mult_pred_same_shape(self, expr: Union[Stack, Concatenate]) -> Tuple[ArraysT,
                                                                              AxesT]:

        one_inst_in_shape = expr.arrays[0].shape
        new_arrays = tuple(self.rec(arr) for arr in expr.arrays)

        _, cur_studies, studies_by_array = self._studies_from_multiple_pred(new_arrays)

        study_to_axis_number: Dict[ParameterStudyAxisTag, int] = {}

        new_shape_of_predecessors = one_inst_in_shape
        new_axes  = expr.axes

        for study in cur_studies:
            if isinstance(study, ParameterStudyAxisTag):
                # Just defensive programming
                # The active studies are added to the end of the bindings.
                study_to_axis_number[study] = len(new_shape_of_predecessors)
                new_shape_of_predecessors = *new_shape_of_predecessors, \
                                            (study.axis_size,)
                new_axes = *new_axes, Axis(tags=frozenset((study,))),
                #  This assumes that the axis only has 1 tag,
                #  because there should be no dependence across instances.

        # This is going to be expensive.

        # Now we need to update the expressions.
        # Now that we have the appropriate shape,
        # we need to update the input arrays to match.

        cp_map = CopyMapper()
        corrected_new_arrays: ArraysT = ()
        for _, array in enumerate(new_arrays):
            tmp = cp_map(array)  # Get a copy of the array.
            if len(array.axes) < len(new_axes):
                # We need to grow the array to the new size.
                for study in cur_studies:
                    if study not in studies_by_array[array]:
                        build: ArraysT = tuple([cp_map(tmp) for
                                                          _ in range(study.axis_size)])
                        tmp = Stack(arrays=build, axis=len(tmp.axes),
                                    axes=(*tmp.axes, Axis(tags=frozenset((study,)))),
                                    tags=tmp.tags,
                                    non_equality_tags=tmp.non_equality_tags)
            elif len(array.axes) > len(new_axes):
                raise ValueError("Input array is too big. " +
                                 f"Expected at most: {len(new_axes)} "  +
                                 f"Found: {len(array.axes)} axes.")

            # Now we need to correct to the appropriate shape with an axis permutation.
            # These are known to be in the right place.
            permute: Tuple[int, ...] = tuple([i for i in range(len(one_inst_in_shape))])

            for _, axis in enumerate(tmp.axes):
                axis_tags = list(axis.tags_of_type(ParameterStudyAxisTag))
                if axis_tags:
                    assert len(axis_tags) == 1
                    permute = *permute, study_to_axis_number[axis_tags[0]],
            assert len(permute) == len(new_shape_of_predecessors)
            corrected_new_arrays = *corrected_new_arrays, \
                                AxisPermutation(tmp, permute, tags=tmp.tags,
                                                 axes=tmp.axes,
                                    non_equality_tags=tmp.non_equality_tags),

        return corrected_new_arrays, new_axes

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        # Update bindings first.
        new_bindings: Dict[str, Array] = {name: self.rec(bnd)
                                             for name, bnd in
                                          sorted(expr.bindings.items())}

        # Determine the new parameter studies that are being conducted.
        from pytools import unique

        all_axis_tags: StudiesT = ()
        varname_to_studies: Dict[str, Dict[UniqueTag, bool]] = {}
        for name, bnd in sorted(new_bindings.items()):
            axis_tags_for_bnd: Set[Tag] = set()
            varname_to_studies[name] = {}
            for i in range(len(bnd.axes)):
                axis_tags_for_bnd = axis_tags_for_bnd.union(bnd.axes[i].tags_of_type(ParameterStudyAxisTag))
            for tag in axis_tags_for_bnd:
                if isinstance(tag, ParameterStudyAxisTag):
                    # Defense
                    varname_to_studies[name][tag] = True
                    all_axis_tags = *all_axis_tags, tag,

        cur_studies: Sequence[ParameterStudyAxisTag] = list(unique(all_axis_tags))
        study_to_axis_number: Dict[ParameterStudyAxisTag, int] = {}

        new_shape = expr.shape
        new_axes  = expr.axes

        for study in cur_studies:
            if isinstance(study, ParameterStudyAxisTag):
                # Just defensive programming
                # The active studies are added to the end of the bindings.
                study_to_axis_number[study] = len(new_shape)
                new_shape = *new_shape, study.axis_size,
                new_axes = *new_axes, Axis(tags=frozenset((study,))),
                #  This assumes that the axis only has 1 tag,
                #  because there should be no dependence across instances.

        # Now we need to update the expressions.
        scalar_expr = ParamAxisExpander()(expr.expr, varname_to_studies,
                                          study_to_axis_number)

        return IndexLambda(expr=scalar_expr,
                           bindings=immutabledict(new_bindings),
                           shape=new_shape,
                           var_to_reduction_descr=expr.var_to_reduction_descr,
                           dtype=expr.dtype,
                           axes=new_axes,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_einsum(self, expr: Einsum) -> Array:

        new_arrays = tuple([self.rec(arg) for arg in expr.args])
        new_axes_for_end, cur_studies, _ = self._studies_from_multiple_pred(new_arrays)

        # Access Descriptors hold the Einsum notation.
        new_access_descriptors = list(expr.access_descriptors)
        study_to_axis_number: Dict[ParameterStudyAxisTag, int] = {}

        new_shape = expr.shape

        for study in cur_studies:
            if isinstance(study, ParameterStudyAxisTag):
                # Just defensive programming
                # The active studies are added to the end.
                study_to_axis_number[study] = len(new_shape)
                new_shape = *new_shape, study.axis_size,

        for ind, array in enumerate(new_arrays):
            for _, axis in enumerate(array.axes):
                axis_tags = list(axis.tags_of_type(ParameterStudyAxisTag))
                if axis_tags:
                    assert len(axis_tags) == 1
                    new_access_descriptors[ind] = new_access_descriptors[ind] + \
                                                (EinsumElementwiseAxis(dim=study_to_axis_number[axis_tags[0]]),)

        return Einsum(tuple(new_access_descriptors), new_arrays,
                     axes=expr.axes + new_axes_for_end,
                     redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                     index_to_access_descr=expr.index_to_access_descr,
                     tags=expr.tags,
                     non_equality_tags=expr.non_equality_tags)

    # }}} Operations with multiple predecessors.


class ParamAxisExpander(IdentityMapper):

    def map_subscript(self, expr: prim.Subscript,
                      varname_to_studies: Mapping[str,
                                                  Mapping[ParameterStudyAxisTag, bool]],
                      study_to_axis_number: Mapping[ParameterStudyAxisTag, int]):
        # We know that we are not changing the variable that we are indexing into.
        # This is stored in the aggregate member of the class Subscript.

        # We only need to modify the indexing which is stored in the index member.
        name = expr.aggregate.name
        if name in varname_to_studies.keys():
            #  These are the single instance information.
            index = self.rec(expr.index, varname_to_studies,
                             study_to_axis_number)

            new_vars: Tuple[prim.Variable, ...] = ()

            for key, num in sorted(study_to_axis_number.items(),
                                   key=lambda item: item[1]):
                if key in varname_to_studies[name]:
                    new_vars = *new_vars, prim.Variable(f"_{num}"),

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

    def transform_loopy_program(self,
                                t_unit: lp.TranslationUnit) -> lp.TranslationUnit:
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

        Before applying :attr:`~ParamStudyLazyPyOpenCLFunctionCaller.f`,
        it is compiled to a :mod:`pytato` DAG that would apply
        :attr:`~ParamStudyLazyPyOpenCLFunctionCaller.f`
        with *args* in a lazy-sense. The intermediary pytato DAG for *args* is
        memoized in *self*.
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

        placeholder_args = [_get_f_placeholder_args_for_param_study(arg, iarg,
                                        input_id_to_name_in_program, self.actx)
                            for iarg, arg in enumerate(args)]
        output_template = self.f(*placeholder_args,
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

        input_shapes = {}
        input_axes = {}
        for key, val in arg_id_to_descr.items():
            if isinstance(val, LeafArrayDescriptor):
                input_shapes[input_id_to_name_in_program[key]] = val.shape
            input_axes[input_id_to_name_in_program[key]] = arg_id_to_arg[key].axes
        expand_map = ExpansionMapper(input_shapes, input_axes)
        # Get the dependencies

        sing_inst_outs = pt.make_dict_of_named_arrays(dict_of_named_arrays)

        # Use the normal compiler now.

        compiled_func = self._dag_to_compiled_func(expand_map(sing_inst_outs),
                                                   # pt_dict_of_named_arrays,
                input_id_to_name_in_program=input_id_to_name_in_program,
                output_id_to_name_in_program=output_id_to_name_in_program,
                output_template=output_template)

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
    update_axes: AxesT = (Axis(tags=frozenset()),)
    for i in range(ndim):
        axis_tags = arg.axes[i].tags_of_type(ParameterStudyAxisTag)
        if not axis_tags:
            update_axes = *update_axes, arg.axes[i],
            newshape.append(arg.shape[i])
    # remove the first one that was placed there for typing.
    update_axes = update_axes[1:]
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
