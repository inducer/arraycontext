from __future__ import annotations


__doc__ = """
.. currentmodule:: arraycontext
.. autoclass:: PyOpenCLArrayContext
.. automodule:: arraycontext.impl.pyopencl.taggable_cl_array
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

from typing import TYPE_CHECKING, Any, Literal, cast, overload
from warnings import warn

import numpy as np
from typing_extensions import Self, override

from arraycontext.container.traversal import (
    rec_map_array_container,
    rec_map_container,
    with_array_context,
)
from arraycontext.context import (
    ArrayContext,
    UntransformedCodeWarning,
)
from arraycontext.typing import (
    Array,
    ArrayOrContainerOrScalar,
    ArrayOrContainerOrScalarT,
    ArrayOrContainerT as ArrayOrContainerT,
    ArrayOrScalar,
    ContainerOrScalarT,
    NumpyOrContainerOrScalar,
    ScalarLike,
    is_scalar_like,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    import loopy as lp
    import pyopencl as cl
    import pyopencl.array as cl_array
    from loopy import TranslationUnit
    from pytools.tag import ToTagSetConvertible

    from arraycontext.impl.pyopencl.taggable_cl_array import TaggableCLArray
    from arraycontext.typing import ArrayContainerT


# {{{ PyOpenCLArrayContext

class PyOpenCLArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :class:`pyopencl.array.Array` instances
    for its base array class.

    .. attribute:: context

        A :class:`pyopencl.Context`.

    .. attribute:: queue

        A :class:`pyopencl.CommandQueue`.

    .. attribute:: allocator

        A PyOpenCL memory allocator. Can also be `None` (default) or `False` to
        use the default allocator. Please note that running with the default
        allocator allocates and deallocates OpenCL buffers directly. If lots
        of arrays are created (e.g. as results of computation), the associated cost
        may become significant. Using e.g. :class:`pyopencl.tools.MemoryPool`
        as the allocator can help avoid this cost.

    .. automethod:: transform_loopy_program
    """

    context: cl.Context
    queue: cl.CommandQueue
    allocator: cl_array.Allocator | None

    _force_device_scalars: Literal[True]
    _passed_force_device_scalars: bool
    _wait_event_queue_length: int

    def __init__(self,
            queue: cl.CommandQueue,
            allocator: cl_array.Allocator | None = None,
            wait_event_queue_length: int | None = None,
            force_device_scalars: bool | None = None) -> None:
        r"""
        :arg wait_event_queue_length: The length of a queue of
            :class:`~pyopencl.Event` objects that are maintained by the
            array context, on a per-kernel-name basis. The events returned
            from kernel execution are appended to the queue, and Once the
            length of the queue exceeds *wait_event_queue_length*, the
            first event in the queue :meth:`pyopencl.Event.wait`\ ed on.

            *wait_event_queue_length* may be set to *False* to disable this feature.

            The use of *wait_event_queue_length* helps avoid enqueuing
            large amounts of work (and, potentially, allocating large amounts
            of memory) far ahead of the actual OpenCL execution front,
            by limiting the number of each type (name, really) of kernel
            that may reside unexecuted in the queue at one time.

        .. note::

            For now, *wait_event_queue_length* should be regarded as an
            experimental feature that may change or disappear at any minute.
        """
        if force_device_scalars is not None:
            warn(
                "`force_device_scalars` is deprecated and will be removed in 2025.",
                DeprecationWarning, stacklevel=2)

            if not force_device_scalars:
                raise ValueError(
                    "Passing force_device_scalars=False is not allowed.")

        import pyopencl as cl
        import pyopencl.array as cl_array

        super().__init__()
        self.context = queue.context
        self.queue = queue
        self.allocator = allocator if allocator else None
        if wait_event_queue_length is None:
            wait_event_queue_length = 10

        self._force_device_scalars = True
        # Subclasses might still be using the old
        # "force_devices_scalars: bool = False" interface, in which case we need
        # to explicitly pass force_device_scalars=True in clone()
        self._passed_force_device_scalars = force_device_scalars is not None

        self._wait_event_queue_length = wait_event_queue_length
        self._kernel_name_to_wait_event_queue: dict[str, list[cl.Event]] = {}

        if queue.device.type & cl.device_type.GPU:
            if allocator is None:
                warn("PyOpenCLArrayContext created without an allocator on a GPU. "
                     "This can lead to high numbers of memory allocations. "
                     "Please consider using a pyopencl.tools.MemoryPool. "
                     "Run with allocator=False to disable this warning.",
                     stacklevel=2)

            if __debug__:
                # Use "running on GPU" as a proxy for "they care about speed".
                warn("You are using the PyOpenCLArrayContext on a GPU, but you "
                        "are running Python in debug mode. Use 'python -O' for "
                        "a noticeable speed improvement.",
                        stacklevel=2)

        self._loopy_transform_cache: \
                dict[lp.TranslationUnit, lp.ExecutorBase] = {}

        # TODO: Ideally this should only be `(TaggableCLArray,)`, but
        # that would break the logic in the downstream users.
        self.array_types = (cl_array.Array,)

    @override
    def _get_fake_numpy_namespace(self):
        from arraycontext.impl.pyopencl.fake_numpy import PyOpenCLFakeNumpyNamespace
        return PyOpenCLFakeNumpyNamespace(self)

    def _rec_map_container(self,
                func: Callable[[Array], Array],
                array: ArrayOrContainerOrScalarT,
                allowed_types: tuple[type, ...] | None = None, *,
                default_scalar: ScalarLike | None = None,
                strict: bool = False
            ) -> ArrayOrContainerOrScalarT:
        import arraycontext.impl.pyopencl.taggable_cl_array as tga

        if allowed_types is None:
            # TODO: replace with 'self.array_types' once `cla.Array` support
            # is completely removed
            allowed_types = (tga.TaggableCLArray,)

        def _wrapper(ary: ArrayOrContainerOrScalar) -> ArrayOrContainerOrScalar:
            if isinstance(ary, allowed_types):
                return func(cast("Array", ary))
            elif not strict and isinstance(ary, self.array_types):
                from warnings import warn
                warn(f"Invoking {type(self).__name__}.{func.__name__[1:]} with "
                    f"{type(ary).__name__} will be unsupported in 2023. Use "
                    "'to_tagged_cl_array' to convert instances to TaggableCLArray.",
                    DeprecationWarning, stacklevel=2)
                return func(tga.to_tagged_cl_array(cast("cl_array.Array", ary)))
            elif is_scalar_like(ary):
                if default_scalar is None:
                    return ary
                else:
                    return np.array(ary).dtype.type(default_scalar)
            else:
                raise TypeError(
                    f"{type(self).__name__}.{func.__name__[1:]} invoked with "
                    f"an unsupported array type: got '{type(ary).__name__}', "
                    f"but expected one of {allowed_types}")

        return cast("ArrayOrContainerOrScalarT",
                rec_map_array_container(_wrapper, array))

    # {{{ ArrayContext interface

    @overload
    def from_numpy(self, array: NDArray[Any]) -> Array:
        ...

    @overload
    def from_numpy(self, array: ScalarLike) -> Array:
        ...

    @overload
    def from_numpy(self, array: ArrayContainerT) -> ArrayContainerT:
        ...

    @override
    def from_numpy(self,
                   array: NumpyOrContainerOrScalar
                   ) -> ArrayOrContainerOrScalar:
        import arraycontext.impl.pyopencl.taggable_cl_array as tga

        def _from_numpy(ary: NDArray[Any]):
            return tga.to_device(self.queue, ary, allocator=self.allocator)

        return with_array_context(
            self._rec_map_container(_from_numpy, array, (np.ndarray,), strict=True),  # pyright: ignore[reportArgumentType]
            actx=self)

    @overload
    def to_numpy(self, array: Array) -> np.ndarray:
        ...

    @overload
    def to_numpy(self, array: ContainerOrScalarT) -> ContainerOrScalarT:
        ...

    @override
    def to_numpy(self,
                 array: ArrayOrContainerOrScalar
                 ) -> NumpyOrContainerOrScalar:
        def _to_numpy(ary):
            return ary.get(queue=self.queue)

        return with_array_context(
            self._rec_map_container(_to_numpy, array),
            actx=None)

    @override
    def freeze(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        def _freeze(ary):
            ary.finish()
            return ary.with_queue(None)

        return with_array_context(self._rec_map_container(_freeze, array), actx=None)

    @override
    def thaw(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        def _thaw(ary):
            return ary.with_queue(self.queue)

        return with_array_context(self._rec_map_container(_thaw, array), actx=self)

    @override
    def tag(self,
            tags: ToTagSetConvertible,
            array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        def _tag(ary):
            return ary.tagged(tags)

        return self._rec_map_container(_tag, array)

    @override
    def tag_axis(self,
                 iaxis: int, tags: ToTagSetConvertible,
                 array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        def _tag_axis(ary: ArrayOrScalar) -> ArrayOrScalar:
            return cast("TaggableCLArray", ary).with_tagged_axis(iaxis, tags)

        return cast("ArrayOrContainerOrScalarT",
                    rec_map_container(_tag_axis, array))

    @override
    def call_loopy(self,
                t_unit: TranslationUnit,
                **kwargs: Any,
            ) -> Mapping[str, TaggableCLArray]:
        try:
            executor = self._loopy_transform_cache[t_unit]
        except KeyError:
            orig_t_unit = t_unit
            executor = self.transform_loopy_program(t_unit).executor(self.context)
            self._loopy_transform_cache[orig_t_unit] = executor
            del orig_t_unit

        evt, result = executor(self.queue, **kwargs, allocator=self.allocator)

        if self._wait_event_queue_length is not False:
            prg_name = executor.t_unit.default_entrypoint.name
            wait_event_queue = self._kernel_name_to_wait_event_queue.setdefault(
                    prg_name, [])

            wait_event_queue.append(evt)
            if len(wait_event_queue) > self._wait_event_queue_length:
                wait_event_queue.pop(0).wait()

        import arraycontext.impl.pyopencl.taggable_cl_array as tga

        # FIXME: Inherit loopy tags for these arrays
        return {name: tga.to_tagged_cl_array(ary) for name, ary in result.items()}

    @override
    def clone(self) -> Self:
        if self._passed_force_device_scalars:
            return type(self)(self.queue, self.allocator,
                    wait_event_queue_length=self._wait_event_queue_length,
                    force_device_scalars=True)
        else:
            return type(self)(self.queue, self.allocator,
                    wait_event_queue_length=self._wait_event_queue_length)

    # }}}

    # {{{ transform_loopy_program

    def transform_loopy_program(self, t_unit: lp.TranslationUnit) -> lp.TranslationUnit:
        from warnings import warn
        warn("Using the base "
                f"{type(self).__name__}.transform_loopy_program "
                "to transform a translation unit. "
                "This is largely a no-op and unlikely to result in fast generated "
                "code."
                f"Instead, subclass {type(self).__name__} and implement "
                "the specific transform logic required to transform the program "
                "for your package or application. Check higher-level packages "
                "(e.g. meshmode), which may already have subclasses you may want "
                "to build on.",
                UntransformedCodeWarning, stacklevel=2)

        import loopy as lp
        default_entrypoint = t_unit.default_entrypoint
        options = default_entrypoint.options
        if not (options.return_dict and options.no_numpy):
            raise ValueError("Loopy kernel passed to call_loopy must "
                    "have return_dict and no_numpy options set. "
                    "Did you use arraycontext.make_loopy_program "
                    "to create this kernel?")

        all_inames = default_entrypoint.all_inames()

        inner_iname = None

        if "i0" in all_inames:
            outer_iname = "i0"

            if "i1" in all_inames:
                inner_iname = "i1"

        else:
            return t_unit

        if inner_iname is not None:
            t_unit = lp.split_iname(t_unit, inner_iname, 16, inner_tag="l.0")
        t_unit = lp.tag_inames(t_unit, {outer_iname: "g.0"})

        return t_unit

    # }}}

    # {{{ properties

    @property
    def permits_inplace_modification(self):
        return True

    @property
    def supports_nonscalar_broadcasting(self):
        return False

    @property
    def permits_advanced_indexing(self):
        return False

    # }}}

# }}}

# vim: foldmethod=marker
