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
import logging

import numpy as np
import pytest


logger = logging.getLogger(__name__)


# {{{ test_pt_actx_key_stringification_uniqueness

def test_pt_actx_key_stringification_uniqueness():
    from arraycontext.impl.pytato.utils import _ary_container_key_stringifier

    assert (_ary_container_key_stringifier(((3, 2), 3))
            != _ary_container_key_stringifier((3, (2, 3))))

    assert (_ary_container_key_stringifier(("tup", 3, "endtup"))
            != _ary_container_key_stringifier(((3,),)))

# }}}


# {{{ test_dataclass_array_container

def test_dataclass_array_container() -> None:
    from dataclasses import dataclass, field
    from typing import Optional

    from arraycontext import dataclass_array_container

    # {{{ string fields

    @dataclass
    class ArrayContainerWithStringTypes:
        x: np.ndarray
        y: "np.ndarray"

    with pytest.raises(TypeError):
        # NOTE: cannot have string annotations in container
        dataclass_array_container(ArrayContainerWithStringTypes)

    # }}}

    # {{{ optional fields

    @dataclass
    class ArrayContainerWithOptional:
        x: np.ndarray
        y: Optional[np.ndarray]

    with pytest.raises(TypeError):
        # NOTE: cannot have wrapped annotations (here by `Optional`)
        dataclass_array_container(ArrayContainerWithOptional)

    # }}}

    # {{{ field(init=False)

    @dataclass
    class ArrayContainerWithInitFalse:
        x: np.ndarray
        y: np.ndarray = field(default_factory=lambda: np.zeros(42),
                              init=False, repr=False)

    with pytest.raises(ValueError):
        # NOTE: init=False fields are not allowed
        dataclass_array_container(ArrayContainerWithInitFalse)

    # }}}

    # {{{ device arrays

    from arraycontext import Array

    @dataclass
    class ArrayContainerWithArray:
        x: Array
        y: Array

    dataclass_array_container(ArrayContainerWithArray)

    # }}}

# }}}


# {{{ test_dataclass_container_unions

def test_dataclass_container_unions() -> None:
    from dataclasses import dataclass
    from typing import Union

    from arraycontext import Array, dataclass_array_container

    # {{{ union fields

    @dataclass
    class ArrayContainerWithUnion:
        x: np.ndarray
        y: Union[np.ndarray, Array]

    dataclass_array_container(ArrayContainerWithUnion)

    # }}}

    # {{{ non-container union

    @dataclass
    class ArrayContainerWithWrongUnion:
        x: np.ndarray
        y: Union[np.ndarray, float]

    with pytest.raises(TypeError):
        # NOTE: float is not an ArrayContainer, so y should fail
        dataclass_array_container(ArrayContainerWithWrongUnion)

    # }}}

# }}}


# {{{ test_stringify_array_container_tree


def test_stringify_array_container_tree() -> None:
    from dataclasses import dataclass

    from arraycontext import (
        Array,
        dataclass_array_container,
        stringify_array_container_tree,
    )

    @dataclass_array_container
    @dataclass(frozen=True)
    class ArrayWrapper:
        ary: Array

    @dataclass_array_container
    @dataclass(frozen=True)
    class SomeContainer:
        points: Array
        radius: float
        centers: ArrayWrapper

    @dataclass_array_container
    @dataclass(frozen=True)
    class SomeOtherContainer:
        disk: SomeContainer
        circle: SomeContainer
        has_disk: bool
        norm_type: str
        extent: float

    rng = np.random.default_rng(seed=42)
    a = ArrayWrapper(ary=rng.random(10))
    d = SomeContainer(points=rng.random((2, 10)), radius=rng.random(), centers=a)
    c = SomeContainer(points=rng.random((2, 10)), radius=rng.random(), centers=a)
    ary = SomeOtherContainer(
        disk=d, circle=c,
        has_disk=True,
        norm_type="l2",
        extent=1)

    logger.info("\n%s", stringify_array_container_tree(ary))

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
