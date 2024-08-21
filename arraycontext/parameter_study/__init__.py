__doc__ = """
.. currentmodule:: arraycontext

A parameter study array context allows a user to pass packed input into his or her single instance program.
These array contexts are derived from the implementations present in :mod:`arraycontext.impl`.
Only :mod:`pytato`-based array contexts have been implemented so far.

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

from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Type,
)

import numpy as np

import loopy as lp
from pytato.array import (
    Array,
    AxesT,
    AxisPermutation,
    ShapeType,
    make_dict_of_named_arrays,
    make_placeholder as make_placeholder,
)
from pytato.transform.parameter_study import (
    ParameterStudyAxisTag,
    ParameterStudyVectorizer,
)
from pytools.tag import Tag, UniqueTag as UniqueTag

from arraycontext import (
    get_container_context_recursively as get_container_context_recursively,
    rec_map_array_container as rec_map_array_container,
    rec_multimap_array_container,
)
from arraycontext.container import (
    ArrayContainer as ArrayContainer,
    deserialize_container as deserialize_container,
    is_array_container_type,
    serialize_container as serialize_container,
    NotAnArrayContainerError,
)
from arraycontext.context import (
    ArrayContext,
    ArrayOrContainerT,
)
from arraycontext.container.traversal import (
    rec_keyed_map_array_container,
    rec_map_reduce_array_container,
)
from arraycontext.impl.pytato import (
    PytatoPyOpenCLArrayContext,
)
from arraycontext.impl.pytato.compile import (
    LazilyPyOpenCLCompilingFunctionCaller,
    LeafArrayDescriptor,
    _ary_container_key_stringifier,
    _get_arg_id_to_arg_and_arg_id_to_descr,
    _to_input_for_compiled,
)


ArraysT = tuple[Array, ...]
StudiesT = tuple[ParameterStudyAxisTag, ...]
ParamStudyTagT = Type[ParameterStudyAxisTag]


import logging


logger = logging.getLogger(__name__)


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
        breakpoint()
        output_template = self.f(*placeholder_args,
                **{kw: _get_f_placeholder_args_for_param_study(arg, kw,
                                               input_id_to_name_in_program,
                                               self.actx)
                    for kw, arg in kwargs.items()})

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

        vectorize = ParameterStudyVectorizer(placeholder_name_to_parameter_studies)
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
    """
    ndim: int = len(arg.shape)
    single_inst_shape: ShapeType = ()
    single_inst_axes: AxesT = ()
    for i in range(ndim):
        axis_tags = arg.axes[i].tags_of_type(ParameterStudyAxisTag)
        if not axis_tags:
            single_inst_axes = (*single_inst_axes, arg.axes[i],)
            single_inst_shape = (*single_inst_shape, arg.shape[i])

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
        breakpoint()
        arg = _to_input_for_compiled(arg, actx)
        return _cut_to_single_instance_size(name, arg)
    elif is_array_container_type(arg.__class__):
        def _rec_to_placeholder(keys, ary):
            name = arg_id_to_name[(kw, *keys)]
            # Transform the DAG to give metadata inference a chance to do its job
            ary = _to_input_for_compiled(ary, actx)
            return _cut_to_single_instance_size(name, ary)

        return rec_keyed_map_array_container(_rec_to_placeholder, arg)
    else:
        raise NotImplementedError(type(arg))


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
        out = out.with_tagged_axis(len(orig_shape),
                                   [study_name_tag_type(len(args))])

        # We have added a new axis.
        assert len(orig_shape) + 1 == len(out.shape)
        # Assert that it has been tagged.
        assert out.axes[-1].tags_of_type(study_name_tag_type)

        return out

    if is_array_container_type(type(args[0])):
        # Need to deal with this as a container.
        # assert isinstance(get_container_context_recursively(args[0]), type(actx))
        # assert isinstance(actx, get_container_context_recursively(args[0]))

        return rec_multimap_array_container(_recursive_stack, *args)

    return _recursive_stack(*args)


from arraycontext.container import (
    serialize_container,
    deserialize_container,
)
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

        breakpoint()
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

        assert study_count == 1

        return out

    def reduce_func(iterable):
        breakpoint()
        return deserialize_container(data, iterable)

    if is_array_container_type(data.__class__):
        # We need to recurse through the system and emit out the indexed arrays.
        breakpoint()
        return rec_map_reduce_array_container(_recursive_split_helper, reduce_func, data)
        return rec_map_array_container(_recursive_split_helper, data)

    return _recursive_split_helper(data)
