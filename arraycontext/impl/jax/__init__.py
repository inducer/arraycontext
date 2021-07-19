"""
.. currentmodule:: arraycontext
.. autoclass:: EagerJAXArrayContext
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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

import numpy as np

from typing import Union, Callable, Any
from pytools.tag import ToTagSetConvertible
from arraycontext.context import ArrayContext, _ScalarLike


class EagerJAXArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses
    :class:`jaxlib.xla_extension.DeviceArrayBase` instances for its base array
    class and performs all array operations eagerly. See
    :class:`~arraycontext.PytatoJAXArrayContext` for a lazier version.

    .. note::

        JAX stores a global configuration state in :data:`jax.config`. Callers
        are expected to maintain those. Most important for scientific computing
        workloads being ``jax_enable_x64``.
    """

    def __init__(self) -> None:
        super().__init__()

        from jax.numpy import DeviceArray
        self.array_types = (DeviceArray, )

    def _get_fake_numpy_namespace(self):
        from .fake_numpy import EagerJAXFakeNumpyNamespace
        return EagerJAXFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    def empty(self, shape, dtype):
        import jax.numpy as jnp
        return jnp.empty(shape=shape, dtype=dtype)

    def zeros(self, shape, dtype):
        import jax.numpy as jnp
        return jnp.zeros(shape=shape, dtype=dtype)

    def from_numpy(self, array: Union[np.ndarray, _ScalarLike]):
        import jax
        return jax.device_put(array)

    def to_numpy(self, array):
        import jax
        # jax.device_get can take scalars as well.
        return jax.device_get(array)

    def call_loopy(self, t_unit, **kwargs):
        raise NotImplementedError("calling loopy on JAX arrays"
                                  " not supported. Maybe rewrite"
                                  " the loopy kernel as numpy-flavored array"
                                  " operations using ArrayContext.np.")

    def freeze(self, array):
        return array.block_until_ready()

    def thaw(self, array):
        return array

    # }}}

    def tag(self, tags: ToTagSetConvertible, array):
        # Sorry, not capable.
        return array

    def tag_axis(self, iaxis, tags: ToTagSetConvertible, array):
        # TODO: See `jax.experiemental.maps.xmap`, proabably that should be useful?
        return array

    def clone(self):
        return type(self)()

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        return f

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        import jax.numpy as jnp
        if arg_names is not None:
            from warnings import warn
            warn("'arg_names' don't bear any significance in "
                 "EagerJAXArrayContext.", stacklevel=2)

        return jnp.einsum(spec, *args)

    @property
    def permits_inplace_modification(self):
        return False

    @property
    def supports_nonscalar_broadcasting(self):
        return True

    @property
    def permits_advanced_indexing(self):
        return True

# vim: foldmethod=marker
