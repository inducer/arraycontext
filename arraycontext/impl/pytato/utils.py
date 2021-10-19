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


from typing import Any, Dict, Set, Tuple, Mapping, FrozenSet, Optional
from pytato.array import SizeParam, Placeholder, make_placeholder, Axis
from pytato.array import Array, DataWrapper, DictOfNamedArrays
from pytato.transform import CopyMapper
from pytools import UniqueNameGenerator
from pytools.tag import Taggable, Tag
import pyopencl.array as cl_array


class TaggableCLArray(cl_array.Array, Taggable):
    """
    A :class:`cl_array.Array` that can be tagged.
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
        self.axes = axes

    def copy(self, queue=cl_array._copy_queue, tags=None, axes=None):
        if tags is not None or axes is not None:
            if queue is not cl_array._copy_queue:
                raise ValueError("Cannot change both 'tags'/'axes' and 'queue'"
                                 " at once.")
            tags = self.tags if tags is None else tags
            axes = self.axes if axes is None else axes
            return self.__class__(None, self.shape, self.dtype,
                                  allocator=self.allocator,
                                  strides=self.strides, data=self.base_data,
                                  offset=self.offset, events=self.events,
                                  _fast=True, _context=self.context,
                                  _queue=self.queue, _size=self.size,
                                  tags=tags, axes=axes)
        else:
            new_with_queue = super().copy(queue=queue)
            return self.__class__(None, new_with_queue.shape,
                                  new_with_queue.dtype,
                                  allocator=new_with_queue.allocator,
                                  strides=new_with_queue.strides,
                                  data=new_with_queue.base_data,
                                  offset=new_with_queue.offset,
                                  events=new_with_queue.events, _fast=True,
                                  _context=new_with_queue.context,
                                  _queue=queue, _size=new_with_queue.size,
                                  tags=new_with_queue.tags,
                                  axes=new_with_queue.axes)


def to_tagged_cl_array(ary: cl_array.Array,
                       axes: Optional[Tuple[Axis, ...]],
                       tags: FrozenSet[Tag]) -> TaggableCLArray:
    """
    Converts a *ary* to a :class:`TaggableCLArray` with *tags* attached to it.

    :arg axes: An instance of :class:`pytato.Axis` for each dimension of the
        array. If passed *None*, then initialized to a :class:`pytato.Axis`
        with no tags attached for each dimension.
    """
    axes = axes if axes is not None else tuple(Axis(frozenset())
                                               for _ in ary.shape)

    return TaggableCLArray(None, ary.shape,
                           ary.dtype,
                           allocator=ary.allocator,
                           strides=ary.strides,
                           data=ary.base_data,
                           offset=ary.offset,
                           events=ary.events, _fast=True,
                           _context=ary.context,
                           _queue=ary.queue, _size=ary.size,
                           tags=tags)


class _DatawrapperToBoundPlaceholderMapper(CopyMapper):
    """
    Helper mapper for :func:`normalize_pt_expr`. Every
    :class:`pytato.DataWrapper` is replaced with a deterministic copy of
    :class:`Placeholder`.
    """
    def __init__(self) -> None:
        super().__init__()
        self.bound_arguments: Dict[str, Any] = {}
        self.vng = UniqueNameGenerator()
        self.seen_inputs: Set[str] = set()

    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        if expr.name is not None:
            if expr.name in self.seen_inputs:
                raise ValueError("Got multiple inputs with the name"
                                 f"{expr.name} => Illegal.")
            self.seen_inputs.add(expr.name)

        # Normalizing names so that we more arrays can have the normalized DAG.
        name = self.vng("_actx_dw")
        self.bound_arguments[name] = expr.data
        return make_placeholder(
                    name=name,
                    shape=tuple(self.rec(s) if isinstance(s, Array) else s
                                for s in expr.shape),
                    dtype=expr.dtype,
                    tags=expr.tags)

    def map_size_param(self, expr: SizeParam) -> Array:
        raise NotImplementedError

    def map_placeholder(self, expr: Placeholder) -> Array:
        raise ValueError("Placeholders cannot appear in"
                         " DatawrapperToBoundPlaceholderMapper.")


def _normalize_pt_expr(expr: DictOfNamedArrays) -> Tuple[DictOfNamedArrays,
                                                         Mapping[str, Any]]:
    """
    Returns ``(normalized_expr, bound_arguments)``.  *normalized_expr* is a
    normalized form of *expr*, with all instances of
    :class:`pytato.DataWrapper` replaced with instances of :class:`Placeholder`
    named in a deterministic manner. The data corresponding to the placeholders
    in *normalized_expr* is recorded in the mapping *bound_arguments*.
    Deterministic naming of placeholders permits more effective caching of
    equivalent graphs.
    """
    normalize_mapper = _DatawrapperToBoundPlaceholderMapper()
    normalized_expr = normalize_mapper(expr)
    return normalized_expr, normalize_mapper.bound_arguments
