"""
.. autoclass:: TaggableCLArray
.. autoclass:: Axis

.. autofunction:: to_tagged_cl_array
"""

import pyopencl.array as cla
from typing import FrozenSet, Union, Sequence, Optional, Tuple
from pytools.tag import Taggable, Tag
from dataclasses import dataclass
from pytools import memoize


@dataclass(frozen=True, eq=True)
class Axis(Taggable):
    """
    Records the tags corresponding to a dimensions of :class:`TaggableCLArray`.
    """
    tags: FrozenSet[Tag]

    def copy(self, **kwargs):
        from dataclasses import replace
        return replace(self, **kwargs)


@memoize
def _construct_untagged_axes(ndim: int) -> Tuple[Axis, ...]:
    return tuple(Axis(frozenset()) for _ in range(ndim))


class TaggableCLArray(cla.Array, Taggable):
    """
    A :class:`pyopencl.array.Array` with additional metadata. This is used by
    :class:`~arraycontext.PytatoPyOpenCLArrayContext` to preserve tags for data
    while frozen, and also in a similar capacity by
    :class:`~arraycontext.PyOpenCLArrayContext`.

    .. attribute:: axes

       A :class:`tuple` of instances of :class:`Axis`, with one :class:`Axis`
       for each dimension of the array.

    .. attribute:: tags

        A :class:`frozenset` of :class:`pytools.tag.Tag`. Typically intended to
        record application-specific metadata to drive the optimizations in
        :meth:`arraycontext.PyOpenCLArrayContext.transform_loopy_program`.
    """
    def __init__(self, cq, shape, dtype, order="C", allocator=None,
                 data=None, offset=0, strides=None, events=None, _flags=None,
                 _fast=False, _size=None, _context=None, _queue=None,
                 axes=None, tags=frozenset()):

        super().__init__(cq=cq, shape=shape, dtype=dtype,
                         order=order, allocator=allocator,
                         data=data, offset=offset,
                         strides=strides, events=events,
                         _flags=_flags, _fast=_fast,
                         _size=_size, _context=_context,
                         _queue=_queue)

        self.tags = tags
        axes = axes if axes is not None else _construct_untagged_axes(len(self
                                                                          .shape))
        self.axes = axes

    def copy(self, queue=cla._copy_queue, tags=None, axes=None, _new_class=None):
        """
        :arg _new_class: The class of the copy. :func:`to_tagged_cl_array` is
            sets this to convert instances of :class:`pyopencl.array.Array` to
            :class:`TaggableCLArray`. If not provided, defaults to
            ``self.__class__``.
        """
        _new_class = self.__class__ if _new_class is None else _new_class

        if queue is not cla._copy_queue:
            # Copying command queue is an involved operation, use super-class'
            # implementation.
            base_instance = super().copy(queue=queue)
        else:
            base_instance = self

        if tags is None and axes is None and _new_class is self.__class__:
            # early exit
            return base_instance

        tags = getattr(base_instance, "tags", frozenset()) if tags is None else tags
        axes = getattr(base_instance, "axes", None) if axes is None else axes

        return _new_class(None,
                          base_instance.shape,
                          base_instance.dtype,
                          allocator=base_instance.allocator,
                          strides=base_instance.strides,
                          data=base_instance.base_data,
                          offset=base_instance.offset,
                          events=base_instance.events, _fast=True,
                          _context=base_instance.context,
                          _queue=base_instance.queue,
                          _size=base_instance.size,
                          tags=tags,
                          axes=axes,
                          )

    def with_tagged_axis(self, iaxis: int,
                         tags: Union[Sequence[Tag], Tag]) -> "TaggableCLArray":
        """
        Returns a copy of *self* with *iaxis*-th axis tagged with *tags*.
        """
        new_axes = (self.axes[:iaxis]
                    + (self.axes[iaxis].tagged(tags),)
                    + self.axes[iaxis+1:])
        return self.copy(axes=new_axes)


def to_tagged_cl_array(ary: cla.Array,
                       axes: Optional[Tuple[Axis, ...]],
                       tags: FrozenSet[Tag]) -> TaggableCLArray:
    """
    Returns a :class:`TaggableCLArray` that is constructed from the data in
    *ary* along with the metadata from *axes* and *tags*.

    :arg axes: An instance of :class:`Axis` for each dimension of the
        array. If passed *None*, then initialized to a :class:`pytato.Axis`
        with no tags attached for each dimension.
    """
    return TaggableCLArray.copy(ary, axes=axes, tags=tags,
                                _new_class=TaggableCLArray)
