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
    from numpy.random import default_rng
    rng = default_rng()

    actx = actx_factory()
    foo = actx.thaw(actx.freeze(
        actx.from_numpy(rng.random((10, 4)))
        .tagged(FooTag())
        .with_tagged_axis(0, BarTag())
        .with_tagged_axis(1, BazTag())
        ))

    assert foo.tags_of_type(FooTag)
    assert foo.axes[0].tags_of_type(BarTag)
    assert foo.axes[1].tags_of_type(BazTag)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
