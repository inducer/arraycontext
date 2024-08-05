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

from typing import Callable, Optional, Tuple

import numpy as np

from pytools.tag import ToTagSetConvertible

from arraycontext.container.traversal import rec_map_array_container, with_array_context
from arraycontext.context import Array, ArrayContext, ArrayOrContainer, ScalarLike


class EagerJAXArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses
    :class:`jax.Array` instances for its base array
    class and performs all array operations eagerly. See
    :class:`~arraycontext.PytatoJAXArrayContext` for a lazier version.

    .. note::

        JAX stores a global configuration state in :data:`jax.config`. Callers
        are expected to maintain those. Most important for scientific computing
        workloads being ``jax_enable_x64``.
    """

    def __init__(self) -> None:
        super().__init__()

        import jax.numpy as jnp
        self.array_types = (jnp.ndarray, )

    def _get_fake_numpy_namespace(self):
        from .fake_numpy import EagerJAXFakeNumpyNamespace
        return EagerJAXFakeNumpyNamespace(self)

    def _rec_map_container(
            self, func: Callable[[Array], Array], array: ArrayOrContainer,
            allowed_types: Optional[Tuple[type, ...]] = None, *,
            default_scalar: Optional[ScalarLike] = None,
            strict: bool = False) -> ArrayOrContainer:
        if allowed_types is None:
            allowed_types = self.array_types

        def _wrapper(ary):
            if isinstance(ary, allowed_types):
                return func(ary)
            elif np.isscalar(ary):
                if default_scalar is None:
                    return ary
                else:
                    return np.array(ary).dtype.type(default_scalar)
            else:
                raise TypeError(
                    f"{type(self).__name__}.{func.__name__[1:]} invoked with "
                    f"an unsupported array type: got '{type(ary).__name__}', "
                    f"but expected one of {allowed_types}")

        return rec_map_array_container(_wrapper, array)

    # {{{ ArrayContext interface

    def empty(self, shape, dtype):
        from warnings import warn
        warn(f"{type(self).__name__}.empty is deprecated and will stop "
            "working in 2023. Prefer actx.np.zeros instead.",
            DeprecationWarning, stacklevel=2)

        import jax.numpy as jnp
        return jnp.empty(shape=shape, dtype=dtype)

    def zeros(self, shape, dtype):
        import jax.numpy as jnp
        return jnp.zeros(shape=shape, dtype=dtype)

    def empty_like(self, ary):
        from warnings import warn
        warn(f"{type(self).__name__}.empty_like is deprecated and will stop "
            "working in 2023. Prefer actx.np.zeros_like instead.",
            DeprecationWarning, stacklevel=2)

        def _empty_like(array):
            return self.empty(array.shape, array.dtype)

        return self._rec_map_container(_empty_like, ary)

    def zeros_like(self, ary):
        from warnings import warn
        warn(f"{type(self).__name__}.zeros_like is deprecated and will stop "
            "working in 2023. Use actx.np.zeros_like instead.",
            DeprecationWarning, stacklevel=2)

        return self.np.zeros_like(ary)

    def from_numpy(self, array):
        def _from_numpy(ary):
            import jax
            return jax.device_put(ary)

        return with_array_context(
            self._rec_map_container(_from_numpy, array, allowed_types=(np.ndarray,)),
            actx=self)

    def to_numpy(self, array):
        def _to_numpy(ary):
            import jax
            return jax.device_get(ary)

        return with_array_context(
            self._rec_map_container(_to_numpy, array),
            actx=None)

    def freeze(self, array):
        def _freeze(ary):
            return ary.block_until_ready()

        return with_array_context(self._rec_map_container(_freeze, array), actx=None)

    def thaw(self, array):
        return with_array_context(array, actx=self)

    def tag(self, tags: ToTagSetConvertible, array):
        # Sorry, not capable.
        return array

    def tag_axis(self, iaxis, tags: ToTagSetConvertible, array):
        # TODO: See `jax.experiemental.maps.xmap`, probably that should be useful?
        return array

    def call_loopy(self, t_unit, **kwargs):
        raise NotImplementedError(
            "Calling loopy on JAX arrays is not supported. Maybe rewrite"
            " the loopy kernel as numpy-flavored array operations using"
            " ArrayContext.np.")

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        import jax.numpy as jnp
        if arg_names is not None:
            from warnings import warn
            warn("'arg_names' don't bear any significance in "
                 f"{type(self).__name__}.", stacklevel=2)

        return jnp.einsum(spec, *args)

    def clone(self):
        return type(self)()

    # }}}

    # {{{ properties

    @property
    def permits_inplace_modification(self):
        return False

    @property
    def supports_nonscalar_broadcasting(self):
        return True

    @property
    def permits_advanced_indexing(self):
        return True

    # }}}

# vim: foldmethod=marker
