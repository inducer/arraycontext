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

    import loopy as lp
    import pyopencl as cl
    import pyopencl.array

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
    import pyopencl as cl
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
