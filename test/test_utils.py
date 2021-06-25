"""Testing for internal  utilities."""


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

from arraycontext import (  # noqa: F401
        pytest_generate_tests_for_array_contexts,
        _acf)

import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    "pyopencl", "pyopencl-deprecated", "pytato-pyopencl"
    ])


def test_pt_actx_key_stringification_uniqueness():
    from arraycontext.impl.pytato.compile import _ary_container_key_stringifier

    assert (_ary_container_key_stringifier(((3, 2), 3))
            != _ary_container_key_stringifier((3, (2, 3))))

    assert (_ary_container_key_stringifier(("tup", 3, "endtup"))
            != _ary_container_key_stringifier(((3,),)))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
