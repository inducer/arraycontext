""" PytatoArrayContext specific tests"""
from __future__ import annotations


__copyright__ = "Copyright (C) 2021 University of Illinois Board of Trustees"

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
import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

import loopy as lp
import pyopencl as cl
import pytato as pt
from pytools.tag import Tag

from arraycontext import (
    ArrayContextFactory,
    PytatoPyOpenCLArrayContext,
    pytest_generate_tests_for_array_contexts,
)
from arraycontext.pytest import _PytestPytatoPyOpenCLArrayContextFactory


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


# {{{ type checking

def verify_is_dag(dag: Any) -> pt.DictOfNamedArrays:
    assert isinstance(dag, pt.DictOfNamedArrays)
    return dag


def verify_is_idx_lambda(ary: Any) -> pt.IndexLambda:
    assert isinstance(ary, pt.IndexLambda)
    return ary


def verify_is_data_wrapper(ary: Any) -> pt.DataWrapper:
    assert isinstance(ary, pt.DataWrapper)
    return ary

# }}}


# {{{ pytato-array context fixture

class _PytatoPyOpenCLArrayContextForTests(PytatoPyOpenCLArrayContext):
    """Like :class:`PytatoPyOpenCLArrayContext`, but applies no program
    transformations whatsoever. Only to be used for testing internal to
    :mod:`arraycontext`.
    """

    def transform_loopy_program(self, t_unit):
        return t_unit


class _PytatoPyOpenCLArrayContextForTestsFactory(
        _PytestPytatoPyOpenCLArrayContextFactory):
    actx_class = _PytatoPyOpenCLArrayContextForTests


pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    _PytatoPyOpenCLArrayContextForTestsFactory,
    ])

# }}}


# {{{ dummy tag types

class FooTag(Tag):
    """
    Foo
    """


class BarTag(Tag):
    """
    Bar
    """


class BazTag(Tag):
    """
    Baz
    """

# }}}


def test_tags_preserved_after_freeze(actx_factory: ArrayContextFactory):
    actx = actx_factory()

    from arraycontext.impl.pytato import _BasePytatoArrayContext
    if not isinstance(actx, _BasePytatoArrayContext):
        pytest.skip("only pytato-based array context are supported")

    from numpy.random import default_rng
    rng = default_rng()

    foo = actx.thaw(actx.freeze(
        actx.from_numpy(rng.random((10, 4)))
        .tagged(FooTag())
        .with_tagged_axis(0, BarTag())
        .with_tagged_axis(1, BazTag())
        ))

    assert foo.tags_of_type(FooTag)
    assert foo.axes[0].tags_of_type(BarTag)
    assert foo.axes[1].tags_of_type(BazTag)


def test_arg_size_limit(actx_factory: Callable[[], PytatoPyOpenCLArrayContext]):
    ran_callback = False

    def my_ctc(what, stage, ir):
        if stage == "final":
            assert ir.target.limit_arg_size_nbytes == 42
            nonlocal ran_callback
            ran_callback = True

    def twice(x):
        return 2 * x

    actx = _PytatoPyOpenCLArrayContextForTests(
        actx_factory().queue, compile_trace_callback=my_ctc, _force_svm_arg_limit=42)

    f = actx.compile(twice)
    f(99)

    assert ran_callback


@pytest.mark.parametrize("pass_allocator", ["auto_none", "auto_true", "auto_false",
                                            "pass_buffer", "pass_svm",
                                            "pass_buffer_pool", "pass_svm_pool"])
def test_pytato_actx_allocator(actx_factory: ArrayContextFactory, pass_allocator):
    base_actx = cast("PytatoPyOpenCLArrayContext", actx_factory())
    alloc = None
    use_memory_pool = None

    if pass_allocator == "auto_none":
        pass
    elif pass_allocator == "auto_true":
        use_memory_pool = True
    elif pass_allocator == "auto_false":
        use_memory_pool = False
    elif pass_allocator == "pass_buffer":
        from pyopencl.tools import ImmediateAllocator
        alloc = ImmediateAllocator(base_actx.queue)
    elif pass_allocator == "pass_svm":
        from pyopencl.characterize import has_coarse_grain_buffer_svm
        if not has_coarse_grain_buffer_svm(base_actx.queue.device):
            pytest.skip("need SVM support for this test")
        from pyopencl.tools import SVMAllocator
        alloc = SVMAllocator(base_actx.queue.context, queue=base_actx.queue)
    elif pass_allocator == "pass_buffer_pool":
        from pyopencl.tools import ImmediateAllocator, MemoryPool
        alloc = MemoryPool(ImmediateAllocator(base_actx.queue))
    elif pass_allocator == "pass_svm_pool":
        from pyopencl.characterize import has_coarse_grain_buffer_svm
        if not has_coarse_grain_buffer_svm(base_actx.queue.device):
            pytest.skip("need SVM support for this test")
        from pyopencl.tools import SVMAllocator, SVMPool
        alloc = SVMPool(SVMAllocator(base_actx.queue.context, queue=base_actx.queue))
    else:
        raise ValueError(f"unknown option {pass_allocator}")

    actx = _PytatoPyOpenCLArrayContextForTests(base_actx.queue, allocator=alloc,
                                               use_memory_pool=use_memory_pool)

    def twice(x):
        return 2 * x

    f = actx.compile(twice)
    res = actx.to_numpy(f(99))

    assert res == 198

    # Also test a case in which SVM is not available
    if pass_allocator in ["auto_none", "auto_true", "auto_false"]:
        from unittest.mock import patch

        with patch("pyopencl.characterize.has_coarse_grain_buffer_svm",
                    return_value=False):
            actx = _PytatoPyOpenCLArrayContextForTests(base_actx.queue,
                        allocator=alloc, use_memory_pool=use_memory_pool)

            from pyopencl.tools import ImmediateAllocator, MemoryPool
            assert isinstance(actx.allocator,
                              MemoryPool if use_memory_pool else ImmediateAllocator)

            f = actx.compile(twice)
            res = actx.to_numpy(f(99))

            assert res == 198


def test_transfer(actx_factory: ArrayContextFactory):
    import numpy as np

    actx = actx_factory()

    # {{{ simple tests

    a = actx.from_numpy(np.array([0, 1, 2, 3])).tagged(FooTag())

    from arraycontext.impl.pyopencl.taggable_cl_array import TaggableCLArray
    assert isinstance(a.data, TaggableCLArray)

    from arraycontext.impl.pytato.utils import transfer_from_numpy, transfer_to_numpy

    ah = transfer_to_numpy(a, actx)
    assert ah != a
    assert a.tags == ah.tags
    assert a.non_equality_tags == ah.non_equality_tags
    assert isinstance(ah.data, np.ndarray)

    with pytest.raises(ValueError):
        _ahh = transfer_to_numpy(ah, actx)

    ad = verify_is_data_wrapper(transfer_from_numpy(ah, actx))
    assert isinstance(ad.data, TaggableCLArray)
    assert ad != ah
    assert ad != a  # copied DataWrappers compare unequal
    assert ad.tags == ah.tags
    assert ad.non_equality_tags == ah.non_equality_tags
    assert np.array_equal(a.data.get(), ad.data.get())

    with pytest.raises(ValueError):
        _add = transfer_from_numpy(ad, actx)

    # }}}

    # {{{ test with DictOfNamedArrays

    dag = pt.make_dict_of_named_arrays({
        "a_expr": a + 2
        })

    dagh = verify_is_dag(transfer_to_numpy(dag, actx))
    assert dagh != dag
    bndh = verify_is_data_wrapper(
        verify_is_idx_lambda(
            dagh["a_expr"].expr).bindings["_in0"])
    assert isinstance(bndh.data, np.ndarray)

    daghd = verify_is_dag(transfer_from_numpy(dagh, actx))
    bndhd = verify_is_data_wrapper(
        verify_is_idx_lambda(
            daghd["a_expr"].expr).bindings["_in0"])
    assert isinstance(bndhd.data, TaggableCLArray)

    # }}}


def test_pass_args_compiled_func(
            actx_factory: Callable[[], PytatoPyOpenCLArrayContext]):
    import numpy as np

    def twice(x, y, a):
        return 2 * x * y * a

    actx = _PytatoPyOpenCLArrayContextForTests(actx_factory().queue)

    dev_scalar = pt.make_data_wrapper(cl.array.to_device(actx.queue, np.float64(23)))

    f = actx.compile(twice)

    assert actx.to_numpy(f(99.0, np.float64(2.0), dev_scalar)) == 2*23*99*2

    compiled_func, = f.program_cache.values()
    ep = compiled_func.pytato_program.program.t_unit.default_entrypoint

    assert isinstance(ep.arg_dict["_actx_in_0"], lp.ValueArg)
    assert isinstance(ep.arg_dict["_actx_in_1"], lp.ValueArg)
    assert isinstance(ep.arg_dict["_actx_in_2"], lp.ArrayArg)


def test_profiling_actx():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)

    actx = PytatoPyOpenCLArrayContext(queue, profile_kernels=True)

    def twice(x):
        return 2 * x

    # {{{ Compiled test

    f = actx.compile(twice)

    assert len(actx._profile_events) == 0

    for _ in range(10):
        assert actx.to_numpy(f(99)) == 198

    assert len(actx._profile_events) == 10
    actx._wait_and_transfer_profile_events()
    assert len(actx._profile_events) == 0
    assert len(actx._profile_results) == 1
    assert len(actx._profile_results["twice"]) == 10

    from arraycontext.impl.pytato.utils import tabulate_profiling_data

    print(tabulate_profiling_data(actx))
    assert len(actx._profile_results) == 0

    # }}}

    # {{{ Uncompiled/frozen test

    assert len(actx._profile_events) == 0

    for _ in range(10):
        assert np.all(actx.to_numpy(twice(actx.from_numpy(np.array([99, 99])))) == 198)

    assert len(actx._profile_events) == 10
    actx._wait_and_transfer_profile_events()
    assert len(actx._profile_events) == 0
    assert len(actx._profile_results) == 1
    assert len(actx._profile_results["frozen_result"]) == 10

    print(tabulate_profiling_data(actx))

    assert len(actx._profile_results) == 0

    # }}}

    # {{{ test disabling profiling

    actx._enable_profiling(False)

    assert len(actx._profile_events) == 0

    for _ in range(10):
        assert actx.to_numpy(f(99)) == 198

    assert len(actx._profile_events) == 0
    assert len(actx._profile_results) == 0

    # }}}

    # {{{ test enabling profiling

    actx._enable_profiling(True)

    assert len(actx._profile_events) == 0

    for _ in range(10):
        assert actx.to_numpy(f(99)) == 198

    assert len(actx._profile_events) == 10
    actx._wait_and_transfer_profile_events()
    assert len(actx._profile_events) == 0
    assert len(actx._profile_results) == 1

    # }}}

    queue2 = cl.CommandQueue(cl_ctx)

    with pytest.raises(RuntimeError):
        PytatoPyOpenCLArrayContext(queue2, profile_kernels=True)

    actx2 = PytatoPyOpenCLArrayContext(queue2)

    with pytest.raises(RuntimeError):
        actx2._enable_profiling(True)


def test_parallelize_disjoint_loop_sets_scalar():
    from loopy.kernel.data import GroupInameTag, LocalInameTag

    from arraycontext.impl.pytato.parallelize import (
        parallelize_disjoint_loop_sets,
    )

    # Scalars only, nothing to parallelize
    t_unit = lp.make_kernel(
            "{:}",
            "out = a + 1",
            [
                lp.GlobalArg("a,out", np.float32, shape=()),
                ...,
            ])

    t_unit = parallelize_disjoint_loop_sets(t_unit, max_device_compute_units=4)

    knl = t_unit.default_entrypoint
    all_tags = {tag for iname in knl.all_inames()
                for tag in knl.iname_tags(iname)}
    assert not any(isinstance(t, (GroupInameTag, LocalInameTag)) for t in all_tags)


def test_parallelize_disjoint_loop_sets_no_outer_inames():
    from loopy.kernel.data import GroupInameTag, LocalInameTag

    from arraycontext.impl.pytato.parallelize import (
        parallelize_disjoint_loop_sets,
    )

    # No outer inames, nothing to parallelize
    t_unit = lp.make_kernel(
            "{[i, j]: 0<=i,j<n}",
            """
            a[i] = 1
            b[i, j] = 2
            c[j] = 3
            """,
            [
                lp.GlobalArg("a,b,c", np.float32, shape=lp.auto),
                ...,
            ],
            assumptions="n>0")

    t_unit = parallelize_disjoint_loop_sets(t_unit, max_device_compute_units=4)

    knl = t_unit.default_entrypoint
    all_tags = {tag for iname in knl.all_inames()
                for tag in knl.iname_tags(iname)}
    assert not any(isinstance(t, (GroupInameTag, LocalInameTag)) for t in all_tags)


def test_parallelize_disjoint_loop_sets_single_non_redn_iname():
    from loopy.kernel.data import GroupInameTag, LocalInameTag

    from arraycontext.impl.pytato.parallelize import (
        parallelize_disjoint_loop_sets,
    )

    cl_ctx = cl.create_some_context()

    n_par = 345
    n_inner = 5

    # Single non-reduction outer iname (i)
    t_unit = lp.make_kernel(
            f"{{[i, k]: 0<=i<{n_par} and 0<=k<{n_inner}}}",
            """
            out1[i] = 2*a[i]
            out2[i, k] = b[i, k] + a[i]
            """,
            [
                lp.GlobalArg("a,out1", np.float32, shape=(n_par,)),
                lp.GlobalArg("b,out2", np.float32, shape=(n_par, n_inner)),
            ])

    ref_t_unit = t_unit
    t_unit = parallelize_disjoint_loop_sets(t_unit, max_device_compute_units=4)

    knl = t_unit.default_entrypoint
    assert knl.iname_tags_of_type("i_local_zero", LocalInameTag) \
        == {LocalInameTag(0)}
    assert knl.iname_tags_of_type("i_group", GroupInameTag) \
        == {GroupInameTag(0)}
    assert knl.iname_tags_of_type("k", (GroupInameTag, LocalInameTag)) == set()

    lp.auto_test_vs_ref(ref_t_unit, cl_ctx, t_unit)


def test_parallelize_disjoint_loop_sets_multiple_non_redn_inames():
    from loopy.kernel.data import GroupInameTag, LocalInameTag

    from arraycontext.impl.pytato.parallelize import (
        parallelize_disjoint_loop_sets,
    )

    cl_ctx = cl.create_some_context()

    n_par = 345
    n_inner = 5

    # Multiple non-reduction outer inames ({i, j})
    t_unit = lp.make_kernel(
            f"{{[i, j, k]: 0<=i,j<{n_par} and 0<=k<{n_inner}}}",
            """
            out1[i, j] = 2*a[i, j]
            out2[i, j, k] = b[i, j, k] + c[k]
            """,
            [
                lp.GlobalArg("a,out1", np.float32, shape=(n_par, n_par)),
                lp.GlobalArg("b,out2", np.float32, shape=(n_par, n_par, n_inner)),
                lp.GlobalArg("c", np.float32, shape=(n_inner,)),
            ])

    ref_t_unit = t_unit
    t_unit = parallelize_disjoint_loop_sets(t_unit, max_device_compute_units=4)

    knl = t_unit.default_entrypoint
    assert knl.iname_tags_of_type("i_group", GroupInameTag) \
        == {GroupInameTag(0)}
    assert knl.iname_tags_of_type("i_local_one", LocalInameTag) \
        == {LocalInameTag(1)}
    assert knl.iname_tags_of_type("j_local_zero", LocalInameTag) \
        == {LocalInameTag(0)}
    assert knl.iname_tags_of_type("k", (GroupInameTag, LocalInameTag)) == set()

    lp.auto_test_vs_ref(ref_t_unit, cl_ctx, t_unit)


def test_parallelize_disjoint_loop_sets_only_redn_iname():
    from loopy.kernel.data import GroupInameTag, LocalInameTag

    from arraycontext.impl.pytato.parallelize import (
        parallelize_disjoint_loop_sets,
    )

    cl_ctx = cl.create_some_context()

    n_par = 345
    n_inner = 5

    # Reduction outer iname(s) only (doesn't matter if there's one or multiple, only
    # one gets parallelized)
    t_unit = lp.make_kernel(
            f"{{[i, k]: 0<=i<{n_par} and 0<=k<{n_inner}}}",
            """
            out1 = sum(i, a[i])
            out2 = sum(i, sum(k, b[i, k]))
            """,
            [
                lp.GlobalArg("a", np.float32, shape=(n_par,)),
                lp.GlobalArg("b", np.float32, shape=(n_par, n_inner)),
                lp.GlobalArg("out1,out2", np.float32, shape=()),
            ])

    ref_t_unit = t_unit
    t_unit = parallelize_disjoint_loop_sets(t_unit, max_device_compute_units=4)

    knl = t_unit.default_entrypoint
    assert knl.iname_tags_of_type("i_local_zero", LocalInameTag) \
        == {LocalInameTag(0)}
    assert knl.iname_tags_of_type("iprcmpt_i_group", GroupInameTag) \
        == {GroupInameTag(0)}
    assert knl.iname_tags_of_type("k", (GroupInameTag, LocalInameTag)) == set()

    lp.auto_test_vs_ref(ref_t_unit, cl_ctx, t_unit)


def test_parallelize_disjoint_loop_sets_mixed():
    from loopy.kernel.data import GroupInameTag, LocalInameTag

    from arraycontext.impl.pytato.parallelize import (
        parallelize_disjoint_loop_sets,
    )

    cl_ctx = cl.create_some_context()

    n_par = 345
    n_inner = 5

    # Two outer inames: one non-reduction (i), one reduction (j)
    t_unit = lp.make_kernel(
            f"{{[i, j, k]: 0<=i,j<{n_par} and 0<=k<{n_inner}}}",
            """
            out1[i] = sum(j, a[i, j])
            out2[i] = sum(j, sum(k, b[i, j, k]))
            """,
            [
                lp.GlobalArg("a", np.float32, shape=(n_par, n_par)),
                lp.GlobalArg("b", np.float32, shape=(n_par, n_par, n_inner)),
                lp.GlobalArg("out1,out2", np.float32, shape=(n_par,)),
            ])

    ref_t_unit = t_unit
    t_unit = parallelize_disjoint_loop_sets(t_unit, max_device_compute_units=4)

    knl = t_unit.default_entrypoint
    assert knl.iname_tags_of_type("i_group", GroupInameTag) \
        == {GroupInameTag(0)}
    assert knl.iname_tags_of_type("j_local_zero", LocalInameTag) \
        == {LocalInameTag(0)}
    assert knl.iname_tags_of_type("k", (GroupInameTag, LocalInameTag)) == set()

    lp.auto_test_vs_ref(ref_t_unit, cl_ctx, t_unit)


def test_parallelize_disjoint_loop_sets_multiple_independent_loop_sets():
    from loopy.kernel.data import GroupInameTag, LocalInameTag

    from arraycontext.impl.pytato.parallelize import (
        parallelize_disjoint_loop_sets,
    )

    cl_ctx = cl.create_some_context()

    n_par = 345
    n_inner = 5

    # Two independent parallelizable disjoint loop sets ({i, j} and {k, l})
    t_unit = lp.make_kernel(
            f"{{[i, j, k, l, m, p]:"
            f" 0<=i,j,k,l<{n_par} and 0<=m,p<{n_inner}}}",
            """
            out1[i, j] = 2*a[i, j]
            out2[i, j, m] = b[i, j, m] + c[m]
            out3[k] = sum(l, d[k, l])
            out4[k] = sum(l, sum(p, e[k, l, p]))
            """,
            [
                lp.GlobalArg("a,out1", np.float32, shape=(n_par, n_par)),
                lp.GlobalArg(
                    "b,out2", np.float32, shape=(n_par, n_par, n_inner)),
                lp.GlobalArg("c", np.float32, shape=(n_inner,)),
                lp.GlobalArg("d", np.float32, shape=(n_par, n_par)),
                lp.GlobalArg("e", np.float32, shape=(n_par, n_par, n_inner)),
                lp.GlobalArg("out3,out4", np.float32, shape=(n_par,)),
            ])

    ref_t_unit = t_unit
    t_unit = parallelize_disjoint_loop_sets(t_unit, max_device_compute_units=4)

    knl = t_unit.default_entrypoint
    assert knl.iname_tags_of_type("i_group", GroupInameTag) \
        == {GroupInameTag(0)}
    assert knl.iname_tags_of_type("i_local_one", LocalInameTag) \
        == {LocalInameTag(1)}
    assert knl.iname_tags_of_type("j_local_zero", LocalInameTag) \
        == {LocalInameTag(0)}
    assert knl.iname_tags_of_type("m", (GroupInameTag, LocalInameTag)) == set()
    assert knl.iname_tags_of_type("k_group", GroupInameTag) \
        == {GroupInameTag(0)}
    assert knl.iname_tags_of_type("l_local_zero", LocalInameTag) \
        == {LocalInameTag(0)}
    assert knl.iname_tags_of_type("p", (GroupInameTag, LocalInameTag)) == set()

    # Currently, each loop set gets its own call kernel, because they get parallelized
    # independently and the launch configurations may end up being different between
    # them. When running at small scales (where the cost of computation is negligible
    # compared to that of kernel launches), it may be worthwhile to be able to force
    # all of the loop sets to be parallelized according to the same configuration so
    # that they can be merged into a smaller total number of kernels. That's not
    # implemented yet, so for now there's a barrier here.
    gbarriers = [insn for insn in knl.instructions
                 if isinstance(insn, lp.BarrierInstruction)
                 and insn.synchronization_kind == "global"]
    assert len(gbarriers) == 1

    lp.auto_test_vs_ref(ref_t_unit, cl_ctx, t_unit)


def test_parallelize_disjoint_loop_sets_multiple_dependent_loop_sets():
    from loopy.kernel.data import GroupInameTag, LocalInameTag

    from arraycontext.impl.pytato.parallelize import (
        parallelize_disjoint_loop_sets,
    )

    cl_ctx = cl.create_some_context()

    n_par = 345
    n_inner = 5

    # Two parallelizable disjoint loop sets ({i, j} and {k, l}), where the second
    # depends on the first
    t_unit = lp.make_kernel(
            f"{{[i, j, k, l, m, p]:"
            f" 0<=i,j,k,l<{n_par} and 0<=m,p<{n_inner}}}",
            """
            tmp1[i, j] = 2*a[i, j] {id=loopset1insn1}
            tmp2[i, j, m] = b[i, j, m] + c[m] {id=loopset1insn2}
            out1[k] = sum(l, tmp1[k, l]) {id=loopset2insn1}
            out2[k] = sum(l, sum(p, tmp2[k, l, p])) {id=loopset2insn2}
            """,
            [
                lp.GlobalArg("a", np.float32, shape=(n_par, n_par)),
                lp.GlobalArg("b", np.float32, shape=(n_par, n_par, n_inner)),
                lp.GlobalArg("c", np.float32, shape=(n_inner,)),
                lp.GlobalArg("out1,out2", np.float32, shape=(n_par,)),
                lp.TemporaryVariable(
                    "tmp1", np.float32, shape=(n_par, n_par),
                    address_space=lp.AddressSpace.GLOBAL),
                lp.TemporaryVariable(
                    "tmp2", np.float32, shape=(n_par, n_par, n_inner),
                    address_space=lp.AddressSpace.GLOBAL),
            ])

    ref_t_unit = t_unit
    t_unit = parallelize_disjoint_loop_sets(t_unit, max_device_compute_units=4)

    knl = t_unit.default_entrypoint
    assert knl.iname_tags_of_type("i_group", GroupInameTag) \
        == {GroupInameTag(0)}
    assert knl.iname_tags_of_type("i_local_one", LocalInameTag) \
        == {LocalInameTag(1)}
    assert knl.iname_tags_of_type("j_local_zero", LocalInameTag) \
        == {LocalInameTag(0)}
    assert knl.iname_tags_of_type("m", (GroupInameTag, LocalInameTag)) == set()
    assert knl.iname_tags_of_type("k_group", GroupInameTag) \
        == {GroupInameTag(0)}
    assert knl.iname_tags_of_type("l_local_zero", LocalInameTag) \
        == {LocalInameTag(0)}
    assert knl.iname_tags_of_type("p", (GroupInameTag, LocalInameTag)) == set()

    # The second loop set depends on the first, so they execute in separate call
    # kernels separated by a single barrier.
    gbarriers = [insn for insn in knl.instructions
                 if isinstance(insn, lp.BarrierInstruction)
                 and insn.synchronization_kind == "global"]
    assert len(gbarriers) == 1
    gbarrier, = gbarriers
    assert "loopset1insn1" in gbarrier.depends_on
    assert "loopset1insn2" in gbarrier.depends_on
    assert gbarrier.id in knl.id_to_insn["loopset2insn1"].depends_on
    assert gbarrier.id in knl.id_to_insn["loopset2insn2"].depends_on

    lp.auto_test_vs_ref(ref_t_unit, cl_ctx, t_unit)


def test_alias_global_temporaries():
    from arraycontext.impl.pytato.parallelize import alias_global_temporaries

    cl_ctx = cl.create_some_context()

    n = 256

    def global_temp(name: str):
        return lp.TemporaryVariable(
            name, np.float32, shape=(n,), address_space=lp.AddressSpace.GLOBAL)

    # A chain of four call kernels (over i, j, k, l), separated by explicit
    # global barriers, each consuming the global temporary produced by the
    # previous one. The temporaries have these (call-kernel-indexed) live
    # intervals:
    #   tmp1: [0, 1], tmp2: [1, 2], tmp3: [2, 3]
    # so tmp1 is dead by the time tmp3 is born and the two can share storage.
    t_unit = lp.make_kernel(
            f"{{[i, j, k, l]: 0<=i,j,k,l<{n}}}",
            """
            tmp1[i] = a[i] + 1
            ... gbarrier
            tmp2[j] = tmp1[j] * 2
            ... gbarrier
            tmp3[k] = tmp2[k] + 3
            ... gbarrier
            out[l]  = tmp3[l] * 4
            """,
            [
                lp.GlobalArg("a,out", np.float32, shape=(n,)),
                global_temp("tmp1"),
                global_temp("tmp2"),
                global_temp("tmp3"),
            ],
            seq_dependencies=True)

    ref_t_unit = t_unit
    t_unit = alias_global_temporaries(t_unit)

    knl = t_unit.default_entrypoint
    base_storages = {
        name: knl.temporary_variables[name].base_storage
        for name in ("tmp1", "tmp2", "tmp3")}

    # Every global temporary should have been assigned a base storage.
    assert all(bs is not None for bs in base_storages.values())

    # tmp1 and tmp3 have disjoint live intervals (and equal size), so they share
    # base storage; tmp2 is alive in between, so it must be distinct.
    assert base_storages["tmp1"] == base_storages["tmp3"]
    assert base_storages["tmp2"] != base_storages["tmp1"]
    assert len(set(base_storages.values())) == 2

    lp.auto_test_vs_ref(ref_t_unit, cl_ctx, t_unit)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
