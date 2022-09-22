""" PytatoArrayContext specific tests"""

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

from arraycontext import PytatoPyOpenCLArrayContext
from arraycontext import pytest_generate_tests_for_array_contexts
from arraycontext.pytest import _PytestPytatoPyOpenCLArrayContextFactory
from pytools.tag import Tag

import pytest
import logging
logger = logging.getLogger(__name__)


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


def test_tags_preserved_after_freeze(actx_factory):
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


def test_arg_size_limit(actx_factory):
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
def test_pytato_actx_allocator(actx_factory, pass_allocator):
    base_actx = actx_factory()
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
            f = actx.compile(twice)
            res = actx.to_numpy(f(99))

            assert res == 198


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
