"""
.. autoclass:: NameHint
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

import sys
from dataclasses import dataclass
from pytools.tag import Tag, UniqueTag
from warnings import warn


@dataclass(frozen=True)
class NameHint(UniqueTag):
    """A tag acting on arrays or array axes. Express that :attr:`name` is a
    useful starting point in forming an identifier for the tagged object.

    .. attribute:: name

        A string. Must be a valid Python identifier. Not necessarily unique.
    """
    name: str

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError("'name' must be an identifier")


# {{{ deprecation handling

try:
    from meshmode.transform_metadata import FirstAxisIsElementsTag \
            as _FirstAxisIsElementsTag
except ImportError:
    # placeholder in case meshmode is too old to have it.
    class _FirstAxisIsElementsTag(Tag):  # type: ignore[no-redef]
        pass


if sys.version_info >= (3, 7):
    def __getattr__(name):
        if name == "FirstAxisIsElementsTag":
            warn(f"'arraycontext.{name}' is deprecated. "
                    f"Use 'meshmode.transform_metadata.{name}' instead. "
                    f"'arraycontext.{name}' will continue to work until 2022.",
                    DeprecationWarning, stacklevel=2)
            return _FirstAxisIsElementsTag
        else:
            raise AttributeError(name)
else:
    FirstAxisIsElementsTag = _FirstAxisIsElementsTag

# }}}


# vim: foldmethod=marker
