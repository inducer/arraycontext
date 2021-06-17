"""Various (undocumented) helper functions to avoid creating dependencies
on other packages (such as pyopencl or meshmode).
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

from typing import Any


def _is_meshmode_dofarray(x: Any) -> bool:
    try:
        from meshmode.dof_array import DOFArray
    except ImportError:
        return False
    else:
        return isinstance(x, DOFArray)


def _is_pyopencl_array(x: Any) -> bool:
    try:
        import pyopencl.array as cla
    except ImportError:
        return False
    else:
        return isinstance(x, cla.Array)


def _flatten_cl_array(ary):
    import pyopencl.array as cl
    assert isinstance(ary, cl.Array)

    if ary.size == 0:
        # Work around https://github.com/inducer/pyopencl/pull/402
        return ary._new_with_changes(
                data=None, offset=0, shape=(0,), strides=(ary.dtype.itemsize,))
    if ary.flags.f_contiguous:
        return ary.reshape(-1, order="F")
    elif ary.flags.c_contiguous:
        return ary.reshape(-1, order="C")
    else:
        raise ValueError("cannot flatten array "
                f"with strides {ary.strides} of {ary.dtype}")
