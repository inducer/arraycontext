"""
.. currentmodule:: arraycontext
.. autoclass:: PytatoArrayContext
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


from arraycontext.fake_numpy import \
        BaseFakeNumpyNamespace, BaseFakeNumpyLinalgNamespace
from arraycontext.context import ArrayContext
import numpy as np
from typing import Any, Callable, Tuple, Union, Sequence
from pytools.tag import Tag
from numbers import Number
import loopy as lp


class _PytatoFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    def norm(self, ary, ord=None):
        # FIXME: handle isinstance(ary, DOFArray) case
        return super().norm(ary, ord)


class _PytatoFakeNumpyNamespace(BaseFakeNumpyNamespace):
    def _get_fake_numpy_linalg_namespace(self):
        return _PytatoFakeNumpyLinalgNamespace(self._array_context)

    @property
    def ns(self):
        return self._array_context.ns

    def exp(self, x):
        import pytato as pt
        from meshmode.dof_array import obj_or_dof_array_vectorize
        return obj_or_dof_array_vectorize(pt.exp, x)

    def sin(self, x):
        import pytato as pt
        from meshmode.dof_array import obj_or_dof_array_vectorize
        return obj_or_dof_array_vectorize(pt.sin, x)

    def reshape(self, a, newshape):
        import pytato as pt

        from meshmode.dof_array import obj_or_dof_array_vectorize_n_args
        return obj_or_dof_array_vectorize_n_args(pt.reshape, a, newshape)

    def transpose(self, a, axes=None):
        import pytato as pt

        from meshmode.dof_array import obj_or_dof_array_vectorize_n_args
        return obj_or_dof_array_vectorize_n_args(pt.transpose, a, axes)

    def concatenate(self, arrays, axis=0):
        import pytato as pt
        from meshmode.dof_array import obj_or_dof_array_vectorize_n_args
        return obj_or_dof_array_vectorize_n_args(pt.concatenate, arrays, axis)

    def ones_like(self, ary):
        def _ones_like(subary):
            import pytato as pt
            return pt.ones(subary.shape, subary.dtype)

        return self._new_like(ary, _ones_like)

    def maximum(self, x, y):
        import pytato as pt
        from meshmode.dof_array import obj_or_dof_array_vectorize_n_args
        return obj_or_dof_array_vectorize_n_args(pt.maximum, x, y)

    def minimum(self, x, y):
        import pytato as pt
        from meshmode.dof_array import obj_or_dof_array_vectorize_n_args
        return obj_or_dof_array_vectorize_n_args(pt.minimum, x, y)

    def where(self, criterion, then, else_):
        # FIXME: where() does not work
        import pytato as pt
        from meshmode.dof_array import obj_or_dof_array_vectorize_n_args
        return obj_or_dof_array_vectorize_n_args(pt.where, criterion, then, else_)

    def sum(self, a, dtype=None):
        import pytato as pt
        if dtype not in [a.dtype, None]:
            raise NotImplementedError
        return pt.sum(a)

    def min(self, a):
        import pytato as pt
        return pt.amin(a)

    def max(self, a):
        import pytato as pt
        return pt.amax(a)

    def stack(self, arrays, axis=0):
        import pytato as pt
        from meshmode.dof_array import obj_or_dof_array_vectorize_n_args
        return obj_or_dof_array_vectorize_n_args(pt.stack, arrays, axis)


class PytatoCompiledOperator:
    def __init__(self, actx, pytato_program, input_spec, output_spec):
        self.actx = actx
        self.pytato_program = pytato_program
        self.input_spec = input_spec
        self.output_spec = output_spec

    def __call__(self, *args):
        import pytato as pt
        import pyopencl.array as cla
        from meshmode.dof_array import DOFArray
        from pytools.obj_array import flat_obj_array

        updated_kwargs = {}

        def from_obj_array_to_input_dict(array, pos):
            input_dict = {}
            for i in range(len(self.input_spec[pos])):
                for j in range(self.input_spec[pos][i]):
                    ary = array[i][j]
                    arg_name = f"_msh_inp_{pos}_{i}_{j}"
                    if arg_name not in (
                            self.pytato_program.program["_pt_kernel"].arg_dict):
                        continue
                    if isinstance(ary, pt.array.DataWrapper):
                        input_dict[arg_name] = ary.data
                    elif isinstance(ary, cla.Array):
                        input_dict[arg_name] = ary
                    elif isinstance(ary, pt.Array):
                        input_dict[arg_name] = self.actx.freeze(
                                ary).with_queue(self.actx.queue)
                    else:
                        raise TypeError("Expect pt.DataWrapper or CL-array, got "
                                f"{type(ary)}")

            return input_dict

        def from_return_dict_to_obj_array(return_dict):
            return flat_obj_array([DOFArray.from_list(self.actx,
                [self.actx.thaw(return_dict[f"_msh_out_{i}_{j}"])
                 for j in range(self.output_spec[i])])
                for i in range(len(self.output_spec))])

        for iarg, arg in enumerate(args):
            if isinstance(arg, np.number):
                arg_name = f"_msh_inp_{iarg}"
                if arg_name not in (
                        self.pytato_program.program["_pt_kernel"].arg_dict):
                    continue

                updated_kwargs[arg_name] = cla.to_device(self.actx.queue,
                        np.array(arg))
            elif isinstance(arg, np.ndarray) and all(isinstance(el, DOFArray)
                                                     for el in arg):
                updated_kwargs.update(from_obj_array_to_input_dict(arg, iarg))
            else:
                raise NotImplementedError("PytatoCompiledOperator cannot handle"
                                          f" '{type(arg)}'s")

        evt, out_dict = self.pytato_program(queue=self.actx.queue,
                                            allocator=self.actx.allocator,
                                            **updated_kwargs)
        evt.wait()

        return from_return_dict_to_obj_array(out_dict)


class PytatoArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :mod:`pytato` data types to represent
    the DOF arrays targeting OpenCL for offloading operations.

    .. attribute:: context

        A :class:`pyopencl.Context`.

    .. attribute:: queue

        A :class:`pyopencl.CommandQueue`.
    """
    import pytato as pt
    _array_type_ = pt.Array

    def __init__(self, queue, allocator=None):
        super().__init__()
        self.queue = queue
        self.allocator = allocator
        self.np = self._get_fake_numpy_namespace()

    def _get_fake_numpy_namespace(self):
        return _PytatoFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    def empty(self, shape, dtype):
        raise ValueError("PytatoArrayContext does not support empty")

    def symbolic_array_var(self, shape, dtype, name=None):
        import pytato as pt
        return pt.make_placeholder(shape=shape, dtype=dtype, name=name)

    def zeros(self, shape, dtype):
        import pytato as pt
        return pt.zeros(shape, dtype)

    def from_numpy(self, np_array: np.ndarray):
        import pytato as pt
        import pyopencl.array as cla
        cl_array = cla.to_device(self.queue, np_array)
        return pt.make_data_wrapper(cl_array)

    def to_numpy(self, array):
        cl_array = self.freeze(array)
        return cl_array.get(queue=self.queue)

    def call_loopy(self, program, **kwargs):
        from pytato.loopy import call_loopy
        import pyopencl.array as cla
        entrypoint, = set(program.callables_table)

        # thaw frozen arrays
        kwargs = {kw: (self.thaw(arg) if isinstance(arg, cla.Array) else arg)
                  for kw, arg in kwargs.items()}

        return call_loopy(program, kwargs, entrypoint)

    def freeze(self, array):
        import pytato as pt
        import pyopencl.array as cla

        if isinstance(array, pt.Placeholder):
            raise ValueError("freezing placeholder would return garbage valued"
                    " arrays")
        if isinstance(array, cla.Array):
            return array.with_queue(None)
        if not isinstance(array, pt.Array):
            raise TypeError("PytatoArrayContext.freeze invoked with non-pt arrays")

        prg = pt.generate_loopy(array, cl_device=self.queue.device)
        evt, (cl_array,) = prg(self.queue)
        evt.wait()

        return cl_array.with_queue(None)

    def thaw(self, array):
        import pytato as pt
        import pyopencl.array as cla

        if not isinstance(array, cla.Array):
            raise TypeError("PytatoArrayContext.thaw expects CL arrays, got "
                    f"{type(array)}")

        return pt.make_data_wrapper(array.with_queue(self.queue))

    # }}}

    def compile(self, f: Callable[[Any], Any],
            inputs_like: Tuple[Union[Number, np.array], ...]) -> Callable[..., Any]:
        from pytools.obj_array import flat_obj_array
        from meshmode.dof_array import DOFArray
        import pytato as pt

        def make_placeholder_like(input_like, pos):
            if isinstance(input_like, np.number):
                return pt.make_placeholder(input_like.dtype,
                                           f"_msh_inp_{pos}")
            elif isinstance(input_like, np.ndarray) and all(isinstance(e, DOFArray)
                                                            for e in input_like):
                return flat_obj_array([DOFArray.from_list(self,
                    [pt.make_placeholder(grp_ary.shape,
                                         grp_ary.dtype, f"_msh_inp_{pos}_{i}_{j}")
                     for j, grp_ary in enumerate(dof_ary)])
                    for i, dof_ary in enumerate(input_like)])

            raise NotImplementedError(f"Unknown input type '{type(input_like)}'.")

        def as_dict_of_named_arrays(fields_obj_ary):
            dict_of_named_arrays = {}
            # output_spec: a list of length #fields; ith-entry denotes #groups in
            # ith-field
            output_spec = []
            for i, field in enumerate(fields_obj_ary):
                output_spec.append(len(field))
                for j, grp in enumerate(field):
                    dict_of_named_arrays[f"_msh_out_{i}_{j}"] = grp

            return pt.make_dict_of_named_arrays(dict_of_named_arrays), output_spec

        outputs = f(*[make_placeholder_like(el, iel)
                      for iel, el in enumerate(inputs_like)])

        if not (isinstance(outputs, np.ndarray)
                and all(isinstance(e, DOFArray)
                        for e in outputs)):
            raise TypeError("Can only pass in functions that return numpy"
                            " array of DOFArrays.")

        output_dict_of_named_arrays, output_spec = as_dict_of_named_arrays(outputs)

        pytato_program = pt.generate_loopy(output_dict_of_named_arrays,
                                           options={"return_dict": True},
                                           cl_device=self.queue.device)

        if False:
            from time import time
            start = time()
            # transforming leads to compile-time slow downs (turning off for now)
            pytato_program.program = self.transform_loopy_program(
                    pytato_program.program)
            end = time()
            print(f"Transforming took {end-start} secs")

        return PytatoCompiledOperator(self, pytato_program,
                                      [[len(arg) for arg in input_like]
                                       if isinstance(input_like, np.ndarray)
                                       else []

                                       for input_like in inputs_like],
                                      output_spec)

    def transform_loopy_program(self, prg):
        from loopy.program import iterate_over_kernels_if_given_program

        nwg = 48
        nwi = (16, 2)

        @iterate_over_kernels_if_given_program
        def gridify(knl):
            # {{{ Pattern matching inames

            for insn in knl.instructions:
                if isinstance(insn, lp.CallInstruction):
                    # must be a callable kernel, don't touch.
                    pass
                elif isinstance(insn, lp.Assignment):
                    bigger_loop = None
                    smaller_loop = None
                    for iname in insn.within_inames:
                        if iname.startswith("iel"):
                            assert bigger_loop is None
                            bigger_loop = iname
                        if iname.startswith("idof"):
                            assert smaller_loop is None
                            smaller_loop = iname

                    if bigger_loop or smaller_loop:
                        assert bigger_loop is not None and smaller_loop is not None
                    else:
                        sorted_inames = sorted(tuple(insn.within_inames),
                                key=knl.get_constant_iname_length)
                        smaller_loop = sorted_inames[0]
                        bigger_loop = sorted_inames[1]

                    knl = lp.chunk_iname(knl, bigger_loop, nwg,
                            outer_tag="g.0")
                    knl = lp.split_iname(knl, f"{bigger_loop}_inner",
                            nwi[0], inner_tag="l.1")
                    knl = lp.split_iname(knl, smaller_loop,
                            nwi[1], inner_tag="l.0")
                elif isinstance(insn, lp.BarrierInstruction):
                    pass
                else:
                    raise NotImplementedError

            # }}}

            return knl

        prg = lp.set_options(prg, "insert_additional_gbarriers")

        return gridify(prg)

    def tag(self, tags: Union[Sequence[Tag], Tag], array):
        # Sorry, not capable.
        return array

    def tag_axis(self, iaxis, tags: Union[Sequence[Tag], Tag], array):
        # Sorry, not capable.
        return array


# }}}
