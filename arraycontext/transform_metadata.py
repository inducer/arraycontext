"""
.. currentmodule:: arraycontext

.. autoclass:: CommonSubexpressionTag
.. autoclass:: ElementwiseMapKernelTag
"""
from __future__ import annotations


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

from pytools.tag import Tag


# {{{ program metadata

class CommonSubexpressionTag(Tag):
    """A tag that is applicable to arrays indicating that this same array
    may be evaluated multiple times, and that the implementation should
    eliminate those redundant evaluations if possible.
    """


class ElementwiseMapKernelTag(Tag):
    """A tag that applies to :class:`loopy.LoopKernel` indicating that the kernel
    is a "map", i.e. that the output array(s) has/have the same shape as the
    input array(s), and that each output element only depends on its corresponding
    element(s) in the input array(s).

    .. note::

        "Element" here refers to a scalar element of an array, not an element
        in a finite-element discretization.
    """

# }}}


# vim: foldmethod=marker
