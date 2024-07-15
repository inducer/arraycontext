""" PytatoArrayContext specific tests on the Parameter Study Module"""

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

import pytest

from pytools.tag import Tag

from arraycontext import (
    PytatoPyOpenCLArrayContext,
    pytest_generate_tests_for_array_contexts,
)
from arraycontext.parameter_study.transform import (
    ParamStudyPytatoPyOpenCLArrayContext,
    ParameterStudyAxisTag
)
from arraycontext.parameter_study import (
        pack_for_parameter_study,
        unpack_parameter_study,
)

from arraycontext.pytest import _PytestPytatoPyOpenCLArrayContextFactory


logger = logging.getLogger(__name__)


# {{{ pytato-array context fixture

class _PytatoPyOpenCLArrayContextForTests(ParamStudyPytatoPyOpenCLArrayContext):
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

class ParamStudy1(ParameterStudyAxisTag):
    """
    1st parameter study.
    """

class ParamStudy2(ParameterStudyAxisTag):
    """
    2bd parameter study.
    """
# }}}


# {{{ Expansion Mapper specific tests.

def test_pack_for_parameter_study(actx_factory):

    actx = actx_factory()

    from arraycontext.impl.pytato import _BasePytatoArrayContext
    if not isinstance(actx, ParamStudyPytatoPyOpenCLArrayContext):
        pytest.skip("only parameter study array contexts are supported")

    import numpy as np
    seed = 12345
    rng = np.random.default_rng(seed)

    base_shape = (15, 5)
    x0 = actx.from_numpy(rng.random(base_shape))
    x1 = actx.from_numpy(rng.random(base_shape))
    x2 = actx.from_numpy(rng.random(base_shape))
    x3 = actx.from_numpy(rng.random(base_shape))
    

    y0 = actx.from_numpy(rng.random(base_shape))
    y1 = actx.from_numpy(rng.random(base_shape))
    y2 = actx.from_numpy(rng.random(base_shape))
    y3 = actx.from_numpy(rng.random(base_shape))
    y4 = actx.from_numpy(rng.random(base_shape))

    def rhs(a,b):
        return a + b

    pack_x = pack_for_parameter_study(actx, ParamStudy1, (4,), x0, x1, x2, x3)
    assert pack_x.shape == (4,15,5)

    pack_y = pack_for_parameter_study(actx, ParamStudy2, (5,), y0,y1, y2,y3,y4)
    assert pack_y.shape == (5,15,5)

    for i in range(3):
        axis_tags = pack_x.axes[i].tags_of_type(ParamStudy1)
        second_tags = pack_x.axes[i].tags_of_type(ParamStudy2)
        if i == 0:
            assert axis_tags
        else:
            assert not axis_tags
        assert not second_tags

def test_unpack_parameter_study(actx_factory):

    actx = actx_factory()

    from arraycontext.impl.pytato import _BasePytatoArrayContext
    if not isinstance(actx, ParamStudyPytatoPyOpenCLArrayContext):
        pytest.skip("only parameter study array contexts are supported")

    import numpy as np
    seed = 12345
    rng = np.random.default_rng(seed)

    base_shape = (15, 5)
    x0 = actx.from_numpy(rng.random(base_shape))
    x1 = actx.from_numpy(rng.random(base_shape))
    x2 = actx.from_numpy(rng.random(base_shape))
    x3 = actx.from_numpy(rng.random(base_shape))
    

    y0 = actx.from_numpy(rng.random(base_shape))
    y1 = actx.from_numpy(rng.random(base_shape))
    y2 = actx.from_numpy(rng.random(base_shape))
    y3 = actx.from_numpy(rng.random(base_shape))
    y4 = actx.from_numpy(rng.random(base_shape))

    def rhs(a,b):
        return a + b

    pack_x = pack_for_parameter_study(actx, ParamStudy1, (4,), x0, x1, x2, x3)
    assert pack_x.shape == (4,15,5)

    pack_y = pack_for_parameter_study(actx, ParamStudy2, (5,), y0,y1, y2,y3,y4)
    assert pack_y.shape == (5,15,5)

    compiled_rhs = actx.compile(rhs)

    output = compiled_rhs(pack_x, pack_y)

    assert output.shape(4,5,15,5)

    output_x = unpack_parameter_study(output, ParamStudy1)
    assert len(output_x) == 1  # Only 1 study associated with this variable.
    assert len(output_x[0]) == 4 # 4 inputs for the parameter study.
    for i in range(len(output_x[0])):
        assert output_x[0][i].shape == (5, 15, 5)


    output_y = unpack_parameter_study(output, ParamStudy2)
    assert len(output_y) == 1  # Only 1 study associated with this variable.
    assert len(output_y[0]) == 5 # 5 inputs for the parameter study.
    for i in range(len(output_y[0])):
        assert output_y[0][i].shape == (4, 15, 5)


# }}}



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
