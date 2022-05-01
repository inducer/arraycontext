"""
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

from warnings import warn
from typing import Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np

from pytools.tag import ToTagSetConvertible

from arraycontext.context import ArrayContext, _ScalarLike
from arraycontext.container.traversal import (rec_map_array_container,
                                              with_array_context)


if TYPE_CHECKING:
    import pyopencl
    import loopy as lp


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

    def __init__(self,
            queue: "pyopencl.CommandQueue",
            allocator: Optional["pyopencl.tools.AllocatorInterface"] = None,
            wait_event_queue_length: Optional[int] = None,
            force_device_scalars: bool = False) -> None:
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

        :arg force_device_scalars: if *True*, scalar results returned from
            reductions in :attr:`ArrayContext.np` will be kept on the device.
            If *False*, the equivalent of :meth:`~ArrayContext.freeze` and
            :meth:`~ArrayContext.to_numpy` is applied to transfer the results
            to the host.
        """
        if not force_device_scalars:
            warn("Configuring the PyOpenCLArrayContext to return host scalars "
                    "from reductions is deprecated. "
                    "To configure the PyOpenCLArrayContext to return "
                    "device scalars, pass 'force_device_scalars=True' to the "
                    "constructor. "
                    "Support for returning host scalars will be removed in 2022.",
                    DeprecationWarning, stacklevel=2)

        import pyopencl as cl
        import pyopencl.array as cl_array

        super().__init__()
        self.context = queue.context
        self.queue = queue
        self.allocator = allocator if allocator else None
        if wait_event_queue_length is None:
            wait_event_queue_length = 10

        self._force_device_scalars = force_device_scalars
        self._wait_event_queue_length = wait_event_queue_length
        self._kernel_name_to_wait_event_queue: Dict[str, List[cl.Event]] = {}

        if queue.device.type & cl.device_type.GPU:
            if allocator is None:
                warn("PyOpenCLArrayContext created without an allocator on a GPU. "
                     "This can lead to high numbers of memory allocations. "
                     "Please consider using a pyopencl.tools.MemoryPool. "
                     "Run with allocator=False to disable this warning.")

            if __debug__:
                # Use "running on GPU" as a proxy for "they care about speed".
                warn("You are using the PyOpenCLArrayContext on a GPU, but you "
                        "are running Python in debug mode. Use 'python -O' for "
                        "a noticeable speed improvement.")

        self._loopy_transform_cache: \
                Dict["lp.TranslationUnit", "lp.TranslationUnit"] = {}

        # TODO: Ideally this should only be `(TaggableCLArray,)`, but
        # that would break the logic in the downstream users.
        self.array_types = (cl_array.Array,)

    def _get_fake_numpy_namespace(self):
        from arraycontext.impl.pyopencl.fake_numpy import PyOpenCLFakeNumpyNamespace
        return PyOpenCLFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    def empty(self, shape, dtype):
        from arraycontext.impl.pyopencl.taggable_cl_array import TaggableCLArray

        return TaggableCLArray(self.queue, shape=shape, dtype=dtype,
                               allocator=self.allocator)

    def zeros(self, shape, dtype):
        import pyopencl.array as cl_array
        from arraycontext.impl.pyopencl.taggable_cl_array import to_tagged_cl_array
        return to_tagged_cl_array(cl_array.zeros(self.queue, shape=shape,
                                                 dtype=dtype,
                                                 allocator=self.allocator),
                                  axes=None, tags=frozenset())

    def from_numpy(self, array: Union[np.ndarray, _ScalarLike]):
        import pyopencl.array as cl_array
        from arraycontext.impl.pyopencl.taggable_cl_array import to_tagged_cl_array
        return to_tagged_cl_array(cl_array
                                  .to_device(self.queue,
                                             array,
                                             allocator=self.allocator),
                                  axes=None, tags=frozenset())

    def to_numpy(self, array):
        if np.isscalar(array):
            return array

        return array.get(queue=self.queue)

    def call_loopy(self, t_unit, **kwargs):
        try:
            t_unit = self._loopy_transform_cache[t_unit]
        except KeyError:
            orig_t_unit = t_unit
            t_unit = self.transform_loopy_program(t_unit)
            self._loopy_transform_cache[orig_t_unit] = t_unit
            del orig_t_unit

        evt, result = t_unit(self.queue, **kwargs, allocator=self.allocator)

        if self._wait_event_queue_length is not False:
            prg_name = t_unit.default_entrypoint.name
            wait_event_queue = self._kernel_name_to_wait_event_queue.setdefault(
                    prg_name, [])

            wait_event_queue.append(evt)
            if len(wait_event_queue) > self._wait_event_queue_length:
                wait_event_queue.pop(0).wait()

        from arraycontext.impl.pyopencl.taggable_cl_array import to_tagged_cl_array
        # FIXME: Inherit loopy tags for these arrays
        return {name: to_tagged_cl_array(ary, axes=None, tags=frozenset())
                for name, ary in result.items()}

    def freeze(self, array):
        import pyopencl.array as cl_array

        def _rec_freeze(ary):
            if isinstance(ary, cl_array.Array):
                ary.finish()
                return ary.with_queue(None)
            else:
                raise TypeError(f"{type(self).__name__} cannot freeze"
                                f" arrays of type '{type(ary).__name__}'.")

        return with_array_context(rec_map_array_container(_rec_freeze, array),
                                  actx=None)

    def thaw(self, array):
        from arraycontext.impl.pyopencl.taggable_cl_array import (TaggableCLArray,
                                                                  to_tagged_cl_array)
        import pyopencl.array as cl_array

        def _rec_thaw(ary):
            if isinstance(ary, TaggableCLArray):
                return ary.with_queue(self.queue)
            elif isinstance(ary, cl_array.Array):
                from warnings import warn
                warn("Invoking PyOpenCLArrayContext.thaw with pyopencl.Array"
                    " will be unsupported in 2023. Use `to_tagged_cl_array`"
                    " to convert instances of pyopencl.Array to TaggableCLArray.",
                    DeprecationWarning, stacklevel=2)
                return (to_tagged_cl_array(ary, axes=None, tags=frozenset())
                        .with_queue(self.queue))
            else:
                raise ValueError("array should be a cl.array.Array,"
                                f" got '{type(ary)}'")

        return with_array_context(rec_map_array_container(_rec_thaw, array),
                                  actx=self)

    # }}}

    def transform_loopy_program(self, t_unit):
        from warnings import warn
        warn("Using arraycontext.PyOpenCLArrayContext.transform_loopy_program "
                "to transform a program. This is deprecated and will stop working "
                "in 2022. Instead, subclass PyOpenCLArrayContext and implement "
                "the specific logic required to transform the program for your "
                "package or application. Check higher-level packages "
                "(e.g. meshmode), which may already have subclasses you may want "
                "to build on.",
                DeprecationWarning, stacklevel=2)

        # accommodate loopy with and without kernel callables

        import loopy as lp
        default_entrypoint = t_unit.default_entrypoint
        options = default_entrypoint.options
        if not (options.return_dict and options.no_numpy):
            raise ValueError("Loopy kernel passed to call_loopy must "
                    "have return_dict and no_numpy options set. "
                    "Did you use arraycontext.make_loopy_program "
                    "to create this kernel?")

        all_inames = default_entrypoint.all_inames()
        # FIXME: This could be much smarter.
        inner_iname = None

        # import with underscore to avoid DeprecationWarning
        from arraycontext.metadata import _FirstAxisIsElementsTag

        if (len(default_entrypoint.instructions) == 1
                and isinstance(default_entrypoint.instructions[0], lp.Assignment)
                and any(isinstance(tag, _FirstAxisIsElementsTag)
                    # FIXME: Firedrake branch lacks kernel tags
                    for tag in getattr(default_entrypoint, "tags", ()))):
            stmt, = default_entrypoint.instructions

            out_inames = [v.name for v in stmt.assignee.index_tuple]
            assert out_inames
            outer_iname = out_inames[0]
            if len(out_inames) >= 2:
                inner_iname = out_inames[1]

        elif "iel" in all_inames:
            outer_iname = "iel"

            if "idof" in all_inames:
                inner_iname = "idof"

        elif "i0" in all_inames:
            outer_iname = "i0"

            if "i1" in all_inames:
                inner_iname = "i1"

        elif not all_inames:
            # no loops, nothing to transform
            return t_unit

        else:
            raise RuntimeError(
                "Unable to reason what outer_iname and inner_iname "
                f"needs to be; all_inames is given as: {all_inames}"
            )

        if inner_iname is not None:
            t_unit = lp.split_iname(t_unit, inner_iname, 16, inner_tag="l.0")
        t_unit = lp.tag_inames(t_unit, {outer_iname: "g.0"})

        return t_unit

    def tag(self, tags: ToTagSetConvertible, array):
        import pyopencl.array as cl_array
        from arraycontext.impl.pyopencl.taggable_cl_array import (TaggableCLArray,
                                                                  to_tagged_cl_array)

        def _rec_tagged(ary):
            if isinstance(ary, TaggableCLArray):
                return ary.tagged(tags)
            elif isinstance(ary, cl_array.Array):
                return to_tagged_cl_array(ary, axes=None,  tags=tags)
            else:
                raise ValueError("array should be a cl.array.Array,"
                                 f" got '{type(ary)}'")

        return rec_map_array_container(_rec_tagged, array)

    def tag_axis(self, iaxis, tags: ToTagSetConvertible, array):
        import pyopencl.array as cl_array
        from arraycontext.impl.pyopencl.taggable_cl_array import (TaggableCLArray,
                                                                  to_tagged_cl_array)

        def _rec_tagged(ary):
            if isinstance(ary, TaggableCLArray):
                return ary.with_tagged_axis(iaxis, tags)
            elif isinstance(ary, cl_array.Array):
                return (to_tagged_cl_array(ary, axes=None,  tags=tags)
                        .with_tagged_axis(iaxis, tags))
            else:
                raise ValueError("array should be a cl.array.Array,"
                                 f" got '{type(ary)}'")

        return rec_map_array_container(_rec_tagged, array)

    def clone(self):
        return type(self)(self.queue, self.allocator,
                wait_event_queue_length=self._wait_event_queue_length,
                force_device_scalars=self._force_device_scalars)

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

# vim: foldmethod=marker
