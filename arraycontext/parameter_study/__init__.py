from __future__ import annotations


__doc__ = """
.. currentmodule:: arraycontext

A parameter study array context allows a user to pass packed input into his or her
single instance program. These array contexts are derived from the implementations
present in :mod:`arraycontext.impl`. Only :mod:`pytato`-based array contexts have
been implemented so far.

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
Copyright (C) 2025-1 University of Illinois Board of Trustees
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

import itertools
from collections.abc import Callable, Mapping
from typing import (
    Any,
)

import numpy as np
from immutabledict import immutabledict

import loopy as lp
from pytato.array import (
    Array,
    AxesT,
    AxisPermutation as AxisPermutation,
    DataWrapper as DataWrapper,
    ParameterStudyDataWrapper,
    ShapeType,
    make_dict_of_named_arrays,
    make_placeholder as make_placeholder,
)
from pytato.transform.parameter_study import (
    ParameterStudyAxisTag,
    ParameterStudyVectorizer,
)
from pytools.tag import Tag as Tag, UniqueTag as UniqueTag

from arraycontext import (
    get_container_context_recursively as get_container_context_recursively,
    rec_map_array_container as rec_map_array_container,
    rec_multimap_array_container,
)
from arraycontext.container import (
    ArrayContainer as ArrayContainer,
    NotAnArrayContainerError as NotAnArrayContainerError,
    deserialize_container as deserialize_container,
    is_array_container_type,
    serialize_container as serialize_container,
)
from arraycontext.container.traversal import (
    rec_keyed_map_array_container,
    rec_map_reduce_array_container as rec_map_reduce_array_container,
    with_array_context,
)
from arraycontext.context import (
    ArrayContext,
    ArrayOrContainerT,
)
from arraycontext.impl.pytato import (
    PytatoPyOpenCLArrayContext,
)
from arraycontext.impl.pytato.compile import (
    AbstractInputDescriptor,
    LazilyPyOpenCLCompilingFunctionCaller,
    LeafArrayDescriptor,
    ScalarInputDescriptor,
    _ary_container_key_stringifier,
    _to_input_for_compiled,
)


ArraysT = tuple[Array, ...]
StudiesT = tuple[ParameterStudyAxisTag, ...]
ParamStudyTagT = type[ParameterStudyAxisTag]


import logging


logger = logging.getLogger(__name__)


def _get_arg_id_to_arg_and_arg_id_to_descr(args: tuple[Any, ...],
                                           kwargs: Mapping[str, Any]
                                           ) -> \
            tuple[Mapping[tuple[Any, ...], Any],
                  Mapping[tuple[Any, ...], AbstractInputDescriptor]]:
    """
    Helper for :meth:`BaseLazilyCompilingFunctionCaller.__call__`. Extracts
    mappings from argument id to argument values and from argument id to
    :class:`AbstractInputDescriptor`. See
    :attr:`CompiledFunction.input_id_to_name_in_program` for argument-id's
    representation.
    """
    arg_id_to_arg: dict[tuple[Any, ...], Any] = {}
    arg_id_to_descr: dict[tuple[Any, ...], AbstractInputDescriptor] = {}

    for kw, arg in itertools.chain(enumerate(args),
                                   kwargs.items()):
        if np.isscalar(arg):
            arg_id = (kw,)
            arg_id_to_arg[arg_id] = arg
            arg_id_to_descr[arg_id] = ScalarInputDescriptor(np.dtype(type(arg)))
        elif is_array_container_type(arg.__class__):
            def id_collector(keys, ary):
                arg_id = (kw, *keys)  # noqa: B023
                if isinstance(ary, ParameterStudyDataWrapper):
                    # Look at the raw data.
                    datawrapper = ary.convert_to_data_wrapper()
                    arg_id_to_arg[arg_id] = datawrapper
                    arg_id_to_descr[arg_id] = LeafArrayDescriptor(np.dtype(datawrapper.dtype),
                                                                  datawrapper.shape)
                    return datawrapper

                else:
                    _placeholder_name_to_parameter_studies = {}
                    _studies_to_size = {}
                    vectorize = ParameterStudyVectorizer(_placeholder_name_to_parameter_studies,
                                                         _studies_to_size)
                    new_ary = vectorize(ary)
                    arg_id_to_arg[arg_id] = new_ary
                    arg_id_to_descr[arg_id] = LeafArrayDescriptor(
                        np.dtype(new_ary.dtype), new_ary.shape)
                return new_ary

            rec_keyed_map_array_container(id_collector, arg)
        elif isinstance(arg, ParameterStudyDataWrapper):
            arg_id = (kw,)
            datawrapper = arg.convert_to_data_wrapper()
            arg_id_to_arg[arg_id] = datawrapper
            arg_id_to_descr[arg_id] = LeafArrayDescriptor(np.dtype(datawrapper.dtype), datawrapper.shape)
        elif isinstance(arg, Array):
            arg_id = (kw,)
            # We need to vectorize the input in case we had something that depended on a parameter study.
            _placeholder_name_to_parameter_studies = {}
            _studies_to_size = {}
            vectorize = ParameterStudyVectorizer(_placeholder_name_to_parameter_studies,
                                             _studies_to_size)
            new_arg = vectorize(arg)
            arg_id_to_arg[arg_id] = new_arg
            arg_id_to_descr[arg_id] = LeafArrayDescriptor(np.dtype(new_arg.dtype),
                                                          new_arg.shape)
        else:
            raise ValueError("Argument to a compiled operator should be"
                             " either a scalar, pt.Array or an array container. Got"
                             f" '{arg}'.")

    return immutabledict(arg_id_to_arg), immutabledict(arg_id_to_descr)


def _cut_to_single_instance_size(name: str, arg: Array) -> Array:
    """
    Helper function to create a placeholder of the single instance size.
    Axes that are removed are those which are marked with a
    :class:`ParameterStudyAxisTag`.

    We need to cut the extra axes off, because we cannot assume that
    the operators we use to build the single instance program will
    understand what to do with the extra axes. We are doing it after the
    call to _to_input_for_compiled in order to ensure that we have an
    :class:`Array` for arg. Also this way we allow the metadata materializer
    to work. See :function:`~arraycontext.impl.pytato._to_input_for_compiled` for more
    information.

    We will return a mapping indicating the pieces we cut off of the placeholder.
    """
    ndim: int = len(arg.shape)
    single_inst_shape: ShapeType = ()
    single_inst_axes: AxesT = ()
    name_to_studies: Mapping[str, StudiesT] = {}
    for i in range(ndim):
        axis_tags = arg.axes[i].tags_of_type(ParameterStudyAxisTag)
        if not axis_tags:
            single_inst_axes = (*single_inst_axes, arg.axes[i],)
            single_inst_shape = (*single_inst_shape, arg.shape[i])
        else:
            tag = next(iter(axis_tags))  # Should only be one tag of this type per axis.
            if name in name_to_studies:
                name_to_studies[name] = (*name_to_studies[name], tag)
            else:
                name_to_studies[name] = (tag)

    return make_placeholder(name, single_inst_shape, arg.dtype, axes=single_inst_axes,
                            tags=arg.tags)

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
        return make_placeholder(name, (), np.dtype(type(arg)))
    elif isinstance(arg, Array):
        name = arg_id_to_name[(kw,)]
        # Transform the DAG to give metadata inference a chance to do its job
        if isinstance(arg, ParameterStudyDataWrapper):
            # Convert to data wrapper so we can cut it appropriately.
            arg = arg.convert_to_data_wrapper()
        arg = _to_input_for_compiled(arg, actx)
        return _cut_to_single_instance_size(name, arg)
    elif is_array_container_type(arg.__class__):
        def _rec_to_placeholder(keys, ary):
            name = arg_id_to_name[(kw, *keys)]
            # Transform the DAG to give metadata inference a chance to do its job
            if isinstance(ary, ParameterStudyDataWrapper):
                ary = ary.convert_to_data_wrapper()
            ary = _to_input_for_compiled(ary, actx)
            return _cut_to_single_instance_size(name, ary)

        return rec_keyed_map_array_container(_rec_to_placeholder, arg)
    else:
        raise NotImplementedError(type(arg))

def _get_f_placeholder_args_for_param_study_and_studies_present(arg,
                    kw, arg_id_to_name, actx) -> tuple[Array, Mapping[str, StudiesT]]:
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
        name = arg_id_to_name[kw,]
        return make_placeholder(name, (), np.dtype(type(arg)))
    elif isinstance(arg, Array):
        name = arg_id_to_name[kw,]
        # Transform the DAG to give metadata inference a chance to do its job
        if isinstance(arg, ParameterStudyDataWrapper):
            # Convert to data wrapper so we can cut it appropriately.
            arg = arg.convert_to_data_wrapper()
        arg = _to_input_for_compiled(arg, actx)
        return _cut_to_single_instance_size(name, arg)
    elif is_array_container_type(arg.__class__):
        def _rec_to_placeholder(keys, ary):
            name = arg_id_to_name[(kw, *keys)]
            # Transform the DAG to give metadata inference a chance to do its job
            if isinstance(ary, ParameterStudyDataWrapper):
                ary = ary.convert_to_data_wrapper()
            ary = _to_input_for_compiled(ary, actx)
            return _cut_to_single_instance_size(name, ary)

        return rec_keyed_map_array_container(_rec_to_placeholder, arg)

    else:
        raise NotImplementedError(type(arg))


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

        placeholder_args = [_get_f_placeholder_args_for_param_study(arg, iarg, # noqa
                                        input_id_to_name_in_program, self.actx)
                            for iarg, arg in enumerate(args)]

        placeholder_kwargs = {kw:
                _get_f_placeholder_args_for_param_study(arg, kw,
                                         input_id_to_name_in_program, self.actx) for
                                          kw, arg in kwargs.items()}

        """
        placeholder_name_to_studies = {}
        ph_args = []
        ph_kwargs = {}
        breakpoint()
        for (arg, mapping_dict) in placeholder_args_and_studies_present:
            ph_args.append(arg)
            placeholder_name_to_studies = placeholder_name_to_studies | mapping_dict

        for (kw, (place, mapping_dict)) in placeholder_kwargs_and_studies.items():
            ph_kwargs[kw] = place
            placeholder_name_to_studies = placeholder_name_to_studies | mapping_dict
        """
        output_template = self.f(*placeholder_args,
                                 **placeholder_kwargs)
        #output_template = self.f(*ph_args, **ph_kwargs)
        self.actx._compile_trace_callback(self.f, "post_trace", output_template)

        if (not (is_array_container_type(output_template.__class__)
                 or isinstance(output_template, Array))):
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

        placeholder_name_to_parameter_studies: dict[str, StudiesT] = {}
        for key, val in arg_id_to_descr.items():
            if isinstance(val, LeafArrayDescriptor):
                name = input_id_to_name_in_program[key]
                for axis in arg_id_to_arg[key].axes:
                    tags = axis.tags_of_type(ParameterStudyAxisTag)
                    if tags:
                        if name in placeholder_name_to_parameter_studies.keys():
                            placeholder_name_to_parameter_studies[name].append(tags)

                        else:
                            placeholder_name_to_parameter_studies[name] = tags
        studies_to_size: dict[ParameterStudyAxisTag, int] = {}
        for studies in placeholder_name_to_parameter_studies.values():
            for study in studies:
                if study in studies_to_size:
                    assert study.size == studies_to_size[study]
                else:
                    studies_to_size[study] = study.size

        vectorize = ParameterStudyVectorizer(placeholder_name_to_parameter_studies,
                                             studies_to_size)
        # Get the dependencies
        sing_inst_outs = make_dict_of_named_arrays(dict_of_named_arrays)

        # Use the normal compiler now.
        compiled_func = self._dag_to_compiled_func(vectorize(sing_inst_outs),
                                                   # pt_dict_of_named_arrays,
                input_id_to_name_in_program=input_id_to_name_in_program,
                output_id_to_name_in_program=output_id_to_name_in_program,
                output_template=output_template)

        self.program_cache[arg_id_to_descr] = compiled_func
        return compiled_func(arg_id_to_arg)

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

    def freeze(self, array):
        if np.isscalar(array):
            return array

        import pyopencl.array as cla
        import pytato as pt

        from arraycontext.container.traversal import rec_keyed_map_array_container
        from arraycontext.impl.pyopencl.taggable_cl_array import (
            TaggableCLArray,
            to_tagged_cl_array,
        )
        from arraycontext.impl.pytato.compile import _ary_container_key_stringifier
        from arraycontext.impl.pytato.utils import (
            _normalize_pt_expr,
            get_cl_axes_from_pt_axes,
        )

        array_as_dict: dict[str, cla.Array | TaggableCLArray | pt.Array] = {}
        key_to_frozen_subary: dict[str, TaggableCLArray] = {}
        key_to_pt_arrays: dict[str, pt.Array] = {}

        def _record_leaf_ary_in_dict(
                key: tuple[Any, ...],
                ary: cla.Array | TaggableCLArray | pt.Array) -> None:
            key_str = "_ary" + _ary_container_key_stringifier(key)
            array_as_dict[key_str] = ary

        rec_keyed_map_array_container(_record_leaf_ary_in_dict, array)

        # {{{ remove any non pytato arrays from array_as_dict

        for key, subary in array_as_dict.items():
            if isinstance(subary, TaggableCLArray):
                key_to_frozen_subary[key] = subary.with_queue(None)
            elif isinstance(subary, self._frozen_array_types):
                from warnings import warn
                warn(f"Invoking {type(self).__name__}.freeze with"
                    f" {type(subary).__name__} will be unsupported in 2023. Use"
                    " `to_tagged_cl_array` to convert instances to TaggableCLArray.",
                    DeprecationWarning, stacklevel=2)

                key_to_frozen_subary[key] = (
                    to_tagged_cl_array(subary.with_queue(None)))
            elif isinstance(subary, pt.array.ParameterStudyDataWrapper):
                # key_to_pt_arrays[key] = subary
                dw = subary.convert_to_data_wrapper()
                key_to_frozen_subary[key] = to_tagged_cl_array(dw.data,
                                                     axes=get_cl_axes_from_pt_axes(dw.axes),
                                                     tags=subary.tags)
            elif isinstance(subary, pt.DataWrapper):
                # trivial freeze.
                key_to_frozen_subary[key] = to_tagged_cl_array(
                    subary.data,
                    axes=get_cl_axes_from_pt_axes(subary.axes),
                    tags=subary.tags)
            elif isinstance(subary, pt.Array):
                # Don't be tempted to take shortcuts here, e.g. for empty
                # arrays, as this will inhibit metadata propagation that
                # may happen in transform_dag below. See
                # https://github.com/inducer/arraycontext/pull/167#issuecomment-1151877480
                key_to_pt_arrays[key] = subary
            else:
                raise TypeError(
                    f"{type(self).__name__}.freeze invoked with an unsupported "
                    f"array type: got '{type(subary).__name__}', but expected one "
                    f"of {self.array_types}")

        # }}}

        def _to_frozen(key: tuple[Any, ...], ary) -> TaggableCLArray:
            key_str = "_ary" + _ary_container_key_stringifier(key)
            return key_to_frozen_subary[key_str]

        if not key_to_pt_arrays:
            # all cl arrays => no need to perform any codegen
            return with_array_context(
                    rec_keyed_map_array_container(_to_frozen, array),
                    actx=None)

        pt_dict_of_named_arrays = pt.make_dict_of_named_arrays(
                key_to_pt_arrays)

        placeholder_name_to_parameter_studies: dict[str, StudiesT] = {}
        studies_to_size: dict[ParameterStudyAxisTag, int] = {}
        # Placeholders are not allowed in _normalize_pt_expr.
        vectorize = ParameterStudyVectorizer(placeholder_name_to_parameter_studies,
                                             studies_to_size)
        pt_dict_of_named_arrays = vectorize(pt_dict_of_named_arrays)
        normalized_expr, bound_arguments = _normalize_pt_expr(
                pt_dict_of_named_arrays)

        try:
            pt_prg = self._freeze_prg_cache[normalized_expr]
        except KeyError:
            try:
                transformed_dag, function_name = (
                        self._dag_transform_cache[normalized_expr])
            except KeyError:
                transformed_dag = self.transform_dag(normalized_expr)

                from pytato.tags import PrefixNamed
                name_hint_tags = []
                for subary in key_to_pt_arrays.values():
                    name_hint_tags.extend(subary.tags_of_type(PrefixNamed))

                from pytools import common_prefix
                name_hint = common_prefix([nh.prefix for nh in name_hint_tags])
                function_name = f"frozen_{name_hint}" if name_hint else "frozen_result"
                from pytato import unify_axes_tags
                transformed_dag = unify_axes_tags(transformed_dag)

                self._dag_transform_cache[normalized_expr] = (
                        transformed_dag, function_name)

            from arraycontext.loopy import _DEFAULT_LOOPY_OPTIONS
            opts = _DEFAULT_LOOPY_OPTIONS
            assert opts.return_dict

            from pytato import unify_axes_tags
            transformed_dag = unify_axes_tags(transformed_dag)

            pt_prg = pt.generate_loopy(transformed_dag,
                                       options=opts,
                                       cl_device=self.queue.device,
                                       function_name=function_name,
                                       target=self.get_target()
                                       ).bind_to_context(self.context)
            pt_prg = pt_prg.with_transformed_translation_unit(
                    self.transform_loopy_program)
            self._freeze_prg_cache[normalized_expr] = pt_prg
        else:
            transformed_dag, function_name = (
                    self._dag_transform_cache[normalized_expr])

        assert len(pt_prg.bound_arguments) == 0
        evt, out_dict = pt_prg(self.queue,
                allocator=self.allocator,
                **bound_arguments)
        evt.wait()
        assert len(set(out_dict) & set(key_to_frozen_subary)) == 0

        key_to_frozen_subary = {
            **key_to_frozen_subary,
            **{k: to_tagged_cl_array(
                    v.with_queue(None),
                    axes=get_cl_axes_from_pt_axes(transformed_dag[k].expr.axes),
                    tags=transformed_dag[k].expr.tags)
               for k, v in out_dict.items()}
        }

        return with_array_context(
                rec_keyed_map_array_container(_to_frozen, array),
                actx=None)

# }}}


def pack_for_parameter_study(actx: ArrayContext,
                             study_name_tag_type: ParamStudyTagT,
                             *args: ArrayOrContainerT) -> ArrayOrContainerT:
    """
        Args is a list of realized input data that needs to be packed
        for a parameter study or uncertainty quantification.

        We assume that each input data set has the same shape and
        are safely castable to the same datatype.
    """

    assert len(args) > 0

    def _recursive_stack(*args: Array) -> Array:
        assert len(args) > 0

        thawed_args: ArraysT = ()
        for val in args:
            assert not is_array_container_type(type(val))
            if not isinstance(val, Array):
                thawed_args = (*thawed_args, actx.thaw(val),)
            else:
                thawed_args = (*thawed_args, val)

        orig_shape = thawed_args[0].shape
        out = actx.np.stack(thawed_args, axis=len(orig_shape))
        out = actx.freeze(out)
        out = out.with_tagged_axis(len(orig_shape), [study_name_tag_type(len(args))])

        # We have added a new axis.
        assert len(orig_shape) + 1 == len(out.shape)

        return ParameterStudyDataWrapper(data=out, shape=orig_shape,
                                        axes=out.axes[:len(orig_shape)],
                                        tags=out.tags,
                                        studies=(study_name_tag_type(len(args)),),
                                        non_equality_tags=frozenset([]))

    if is_array_container_type(type(args[0])):
        # Need to deal with this as a container.
        # assert isinstance(get_container_context_recursively(args[0]), type(actx))
        # assert isinstance(actx, get_container_context_recursively(args[0]))

        return rec_multimap_array_container(_recursive_stack, *args)

    return _recursive_stack(*args)


def unpack_parameter_study(data: ArrayOrContainerT,
                           study_name_tag_type: ParamStudyTagT) -> Mapping[int,
                                                                           ArrayOrContainerT]:
    """
    Recurse through the data structure and split the data along the
    axis which corresponds to the input tag name.
    """

    def _recursive_split_helper(data: Array) -> Mapping[int, Array]:
        """
        Split the data array along the axes which vary according to a
        ParameterStudyAxisTag whose name tag is an instance study_name_tag_type.
        """

        ndim: int = len(data.shape)
        out: list[Array] = []

        study_count = 0
        for i in range(ndim):
            axis_tags = data.axes[i].tags_of_type(study_name_tag_type)
            if axis_tags:
                study_count += 1
                # Now we need to split this data.
                for j in range(data.shape[i]):
                    tmp: list[slice | int] = [slice(None)] * ndim
                    tmp[i] = j
                    the_slice = tuple(tmp)
                    # Needs to be a tuple of slices not list of slices.
                    out.append(data[the_slice])

        assert study_count <= 1

        return out

    def reduce_func(iterable):
        return deserialize_container(data, iterable)

    if is_array_container_type(data.__class__):
        # We need to recurse through the system and emit out the indexed arrays.
        # return rec_map_reduce_array_container(_recursive_split_helper,
        #                                       reduce_func, data)
        return rec_map_array_container(_recursive_split_helper, data)

    return _recursive_split_helper(data)
