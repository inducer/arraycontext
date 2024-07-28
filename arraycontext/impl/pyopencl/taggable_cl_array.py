"""
.. autoclass:: TaggableCLArray
.. autoclass:: Axis

.. autofunction:: to_tagged_cl_array
"""

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

import numpy as np

import pyopencl.array as cla
from pytools import memoize
from pytools.tag import Tag, Taggable, ToTagSetConvertible


# {{{ utils

@dataclass(frozen=True, eq=True)
class Axis(Taggable):
    """
    Records the tags corresponding to a dimension of :class:`TaggableCLArray`.
    """

    tags: FrozenSet[Tag]

    def _with_new_tags(self, tags: FrozenSet[Tag]) -> "Axis":
        from dataclasses import replace
        return replace(self, tags=tags)


@memoize
def _construct_untagged_axes(ndim: int) -> Tuple[Axis, ...]:
    return tuple(Axis(frozenset()) for _ in range(ndim))


def _unwrap_cl_array(ary: cla.Array) -> Dict[str, Any]:
    return {
        "shape": ary.shape,
        "dtype": ary.dtype,
        "allocator": ary.allocator,
        "strides": ary.strides,
        "data": ary.base_data,
        "offset": ary.offset,
        "events": ary.events,
        "_context": ary.context,
        "_queue": ary.queue,
        "_size": ary.size,
        "_fast": True,
        }

# }}}


# {{{ TaggableCLArray

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

        if __debug__:
            if not isinstance(tags, frozenset):
                raise TypeError("tags are not a frozenset")

            if axes is not None and len(axes) != self.ndim:
                raise ValueError("axes length does not match array dimension: "
                                 f"got {len(axes)} axes for {self.ndim}d array")

        if axes is None:
            axes = _construct_untagged_axes(self.ndim)

        self.tags = tags
        self.axes = axes

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype}, "
                f"tags={self.tags}, axes={self.axes})")

    def copy(self, queue=cla._copy_queue):
        ary = super().copy(queue=queue)
        return type(self)(None, tags=self.tags, axes=self.axes,
                          **_unwrap_cl_array(ary))

    def _with_new_tags(self, tags: FrozenSet[Tag]) -> "TaggableCLArray":
        return type(self)(None, tags=tags, axes=self.axes,
                          **_unwrap_cl_array(self))

    def with_tagged_axis(self, iaxis: int,
                         tags: ToTagSetConvertible) -> "TaggableCLArray":
        """
        Returns a copy of *self* with *iaxis*-th axis tagged with *tags*.
        """
        new_axes = (self.axes[:iaxis]
                    + (self.axes[iaxis].tagged(tags),)
                    + self.axes[iaxis+1:])

        return type(self)(None, tags=self.tags, axes=new_axes,
                          **_unwrap_cl_array(self))


def to_tagged_cl_array(ary: cla.Array,
                       axes: Optional[Tuple[Axis, ...]] = None,
                       tags: FrozenSet[Tag] = frozenset()) -> TaggableCLArray:
    """
    Returns a :class:`TaggableCLArray` that is constructed from the data in
    *ary* along with the metadata from *axes* and *tags*. If *ary* is already a
    :class:`TaggableCLArray`, the new *tags* and *axes* are added to the
    existing ones.

    :arg axes: An instance of :class:`Axis` for each dimension of the
        array. If passed *None*, then initialized to a :class:`pytato.Axis`
        with no tags attached for each dimension.
    """
    if axes is not None and len(axes) != ary.ndim:
        raise ValueError("axes length does not match array dimension: "
                         f"got {len(axes)} axes for {ary.ndim}d array")

    from pytools.tag import normalize_tags
    tags = normalize_tags(tags)

    if isinstance(ary, TaggableCLArray):
        if axes is not None:
            for i, axis in enumerate(axes):
                ary = ary.with_tagged_axis(i, axis.tags)

        if tags:
            ary = ary.tagged(tags)

        return ary
    elif isinstance(ary, cla.Array):
        return TaggableCLArray(None, tags=tags, axes=axes,
                               **_unwrap_cl_array(ary))
    else:
        raise TypeError(f"unsupported array type: '{type(ary).__name__}'")

# }}}


# {{{ creation

def empty(queue, shape, dtype=float, *,
        axes: Optional[Tuple[Axis, ...]] = None,
        tags: FrozenSet[Tag] = frozenset(),
        order: str = "C",
        allocator=None) -> TaggableCLArray:
    if dtype is not None:
        dtype = np.dtype(dtype)

    return TaggableCLArray(
        queue, shape, dtype,
        axes=axes, tags=tags,
        order=order, allocator=allocator)


def zeros(queue, shape, dtype=float, *,
        axes: Optional[Tuple[Axis, ...]] = None,
        tags: FrozenSet[Tag] = frozenset(),
        order: str = "C",
        allocator=None) -> TaggableCLArray:
    result = empty(
        queue, shape, dtype=dtype, axes=axes, tags=tags,
        order=order, allocator=allocator)
    result._zero_fill()

    return result


def to_device(queue, ary, *,
        axes: Optional[Tuple[Axis, ...]] = None,
        tags: FrozenSet[Tag] = frozenset(),
        allocator=None):
    return to_tagged_cl_array(
        cla.to_device(queue, ary, allocator=allocator),
        axes=axes, tags=tags)

# }}}
