# mypy: disallow-untyped-defs

"""
.. _freeze-thaw:

Freezing and thawing
--------------------

One of the central concepts introduced by the array context formalism is
the notion of :meth:`~arraycontext.ArrayContext.freeze` and
:meth:`~arraycontext.ArrayContext.thaw`. Each array handled by the array context
is either "thawed" or "frozen". Unlike the real-world concept of freezing and
thawing, these operations leave the original array alone; instead, a semantically
separate array in the desired state is returned.

*   "Thawed" arrays are associated with an array context. They use that context
    to carry out operations (arithmetic, function calls).

*   "Frozen" arrays are static data. They are not associated with an array context,
    and no operations can be performed on them.

Freezing and thawing may be used to move arrays from one array context to another,
as long as both array contexts use identical in-memory data representation.
Otherwise, a common format must be agreed upon, for example using
:mod:`numpy` through :meth:`~arraycontext.ArrayContext.to_numpy` and
:meth:`~arraycontext.ArrayContext.from_numpy`.

.. _freeze-thaw-guidelines:

Usage guidelines
^^^^^^^^^^^^^^^^
Here are some rules of thumb to use when dealing with thawing and freezing:

-   Any array that is stored for a long time needs to be frozen.
    "Memoized" data (cf. :func:`pytools.memoize` and friends) is a good example
    of long-lived data that should be frozen.

-   Within a function, if the user did not supply an array context,
    then any data returned to the user should be frozen.

-   Note that array contexts need not necessarily be passed as a separate
    argument. Passing thawed data as an argument to a function suffices
    to supply an array context. The array context can be extracted from
    a thawed argument using, e.g., :func:`~arraycontext.get_container_context_opt`
    or :func:`~arraycontext.get_container_context_recursively`.

What does this mean concretely?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Freezing and thawing are abstract names for concrete operations. It may be helpful
to understand what these operations mean in the concrete case of various
actual array contexts:

-   Each :class:`~arraycontext.PyOpenCLArrayContext` is associated with a
    :class:`pyopencl.CommandQueue`. In order to operate on array data,
    such a command queue is necessary; it is the main means of synchronization
    between the host program and the compute device. "Thawing" here
    means associating an array with a command queue, and "freezing" means
    ensuring that the array data is fully computed in memory and
    decoupling the array from the command queue. It is not valid to "mix"
    arrays associated with multiple queues within an operation: if it were allowed,
    a dependent operation might begin computing before an input is fully
    available. (Since bugs of this nature would be very difficult to
    find, :class:`pyopencl.array.Array` and
    :class:`~meshmode.dof_array.DOFArray` will not allow them.)

-   For the lazily-evaluating array context based on :mod:`pytato`,
    "thawing" corresponds to the creation of a symbolic "handle"
    (specifically, a :class:`pytato.array.DataWrapper`) representing
    the array that can then be used in computation, and "freezing"
    corresponds to triggering (code generation and) evaluation of
    an array expression that has been built up by the user
    (using, e.g. :func:`pytato.generate_loopy`).


.. currentmodule:: arraycontext

The :class:`ArrayContext` Interface
-----------------------------------

.. autoclass:: ArrayContext

.. autofunction:: tag_axes

Types and Type Variables for Arrays and Containers
--------------------------------------------------

.. autoclass:: Array

.. class:: ArrayT

    A type variable with a lower bound of :class:`Array`.

.. class:: ScalarLike

    A type annotation for scalar types commonly usable with arrays.

See also :class:`ArrayContainer` and :class:`ArrayOrContainerT`.

.. class:: ArrayOrContainer

.. class:: ArrayOrContainerT

    A type variable with a lower bound of :class:`ArrayOrContainer`.

.. class:: ArrayOrContainerOrScalar

.. class:: ArrayOrContainerOrScalarT

    A type variable with a lower bound of :class:`ArrayOrContainerOrScalar`.

Internal typing helpers (do not import)
---------------------------------------

.. currentmodule:: arraycontext.context

This is only here because the documentation tool wants it.

.. class:: SelfType

Canonical locations for type annotations
----------------------------------------

.. class:: ArrayT

    :canonical: arraycontext.ArrayT

.. class:: ArrayOrContainerT

    :canonical: arraycontext.ArrayOrContainerT

.. class:: ArrayOrContainerOrScalarT

    :canonical: arraycontext.ArrayOrContainerOrScalarT
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

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)
from warnings import warn

import numpy as np

from pytools import memoize_method
from pytools.tag import ToTagSetConvertible


if TYPE_CHECKING:
    import loopy

    from arraycontext.container import ArrayContainer


# {{{ typing

ScalarLike = Union[int, float, complex, np.generic]

SelfType = TypeVar("SelfType")


class Array(Protocol):
    """A :class:`~typing.Protocol` for the array type supported by
    :class:`ArrayContext`.

    This is meant to aid in typing annotations. For a explicit list of
    supported types see :attr:`ArrayContext.array_types`.

    .. attribute:: shape
    .. attribute:: size
    .. attribute:: dtype
    .. attribute:: __getitem__
    """

    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def dtype(self) -> "np.dtype[Any]":
        ...

    # Covering all the possible index variations is hard and (kind of) futile.
    # If you'd  like to see how, try changing the Any to
    # AxisIndex = slice | int | "Array"
    # Index = AxisIndex |tuple[AxisIndex]
    def __getitem__(self, index: Any) -> "Array":
        ...


# deprecated, use ScalarLike instead
Scalar = ScalarLike


ArrayT = TypeVar("ArrayT", bound=Array)
ArrayOrContainer = Union[Array, "ArrayContainer"]
ArrayOrContainerT = TypeVar("ArrayOrContainerT", bound=ArrayOrContainer)
ArrayOrContainerOrScalar = Union[Array, "ArrayContainer", ScalarLike]
ArrayOrContainerOrScalarT = TypeVar(
        "ArrayOrContainerOrScalarT",
        bound=ArrayOrContainerOrScalar)

NumpyOrContainerOrScalar = Union[np.ndarray, "ArrayContainer", ScalarLike]

# }}}


# {{{ ArrayContext

class ArrayContext(ABC):
    r"""
    :canonical: arraycontext.ArrayContext

    An interface that allows software implementing a numerical algorithm
    (such as :class:`~meshmode.discretization.Discretization`) to create and interact
    with arrays without knowing their types.

    .. versionadded:: 2020.2

    .. automethod:: from_numpy
    .. automethod:: to_numpy
    .. automethod:: call_loopy
    .. automethod:: einsum
    .. attribute:: np

         Provides access to a namespace that serves as a work-alike to
         :mod:`numpy`. The actual level of functionality provided is up to the
         individual array context implementation, however the functions and
         objects available under this namespace must not behave differently
         from :mod:`numpy`.

         As a baseline, special functions available through :mod:`loopy`
         (e.g. ``sin``, ``exp``) are accessible through this interface.
         A full list of implemented functionality is given in
         :ref:`numpy-coverage`.

         Callables accessible through this namespace vectorize over object
         arrays, including :class:`arraycontext.ArrayContainer`\ s.

    .. attribute:: array_types

        A :class:`tuple` of types that are the valid array classes the
        context can operate on. However, it is not necessary that *all* the
        :class:`ArrayContext`\ 's operations are legal for the types in
        *array_types*. Note that this tuple is *only* intended for use
        with :func:`isinstance`. Other uses are not allowed. This allows
        for 'types' with overridden :meth:`class.__instancecheck__`.

    .. automethod:: freeze
    .. automethod:: thaw
    .. automethod:: freeze_thaw
    .. automethod:: tag
    .. automethod:: tag_axis
    .. automethod:: compile
    .. automethod:: outline
    """

    array_types: Tuple[type, ...] = ()

    def __init__(self) -> None:
        self.np = self._get_fake_numpy_namespace()

    @abstractmethod
    def _get_fake_numpy_namespace(self) -> Any:
        ...

    def __hash__(self) -> int:
        raise TypeError(f"unhashable type: '{type(self).__name__}'")

    def zeros(self,
              shape: Union[int, Tuple[int, ...]],
              dtype: "np.dtype[Any]") -> Array:
        warn(f"{type(self).__name__}.zeros is deprecated and will stop "
            "working in 2025. Use actx.np.zeros instead.",
            DeprecationWarning, stacklevel=2)

        return self.np.zeros(shape, dtype)

    @abstractmethod
    def from_numpy(self,
                   array: NumpyOrContainerOrScalar
                   ) -> ArrayOrContainerOrScalar:
        r"""
        :returns: the :class:`numpy.ndarray` *array* converted to the
            array context's array type. The returned array will be
            :meth:`thaw`\ ed. When working with array containers each leaf
            must be an :class:`~numpy.ndarray` or scalar, which is then converted
            to the context's array type leaving the container structure
            intact.
        """

    @abstractmethod
    def to_numpy(self,
                 array: ArrayOrContainerOrScalar
                 ) -> NumpyOrContainerOrScalar:
        r"""
        :returns: an :class:`numpy.ndarray` for each array recognized by the
            context. The input *array* must be :meth:`thaw`\ ed.
            When working with array containers each leaf must be one of
            the context's array types or a scalar, which is then converted to
            an :class:`~numpy.ndarray` leaving the container structure intact.
        """

    @abstractmethod
    def call_loopy(self,
                   t_unit: "loopy.TranslationUnit",
                   **kwargs: Any) -> Dict[str, Array]:
        """Execute the :mod:`loopy` program *program* on the arguments
        *kwargs*.

        *program* is a :class:`loopy.LoopKernel` or :class:`loopy.TranslationUnit`.
        It is expected to not yet be transformed for execution speed.
        It must have :attr:`loopy.Options.return_dict` set.

        :return: a :class:`dict` of outputs from the program, each an
            array understood by the context.
        """

    @abstractmethod
    def freeze(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        """Return a version of the context-defined array *array* that is
        'frozen', i.e. suitable for long-term storage and reuse. Frozen arrays
        do not support arithmetic. For example, in the context of
        :class:`~pyopencl.array.Array`, this might mean stripping the array
        of an associated command queue, whereas in a lazily-evaluated context,
        it might mean that the array is evaluated and stored.

        Freezing makes the array independent of this :class:`ArrayContext`;
        it is permitted to :meth:`thaw` it in a different one, as long as that
        context understands the array format.
        """

    @abstractmethod
    def thaw(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        """Take a 'frozen' array and return a new array representing the data in
        *array* that is able to perform arithmetic and other operations, using
        the execution resources of this context. In the context of
        :class:`~pyopencl.array.Array`, this might mean that the array is
        equipped with a command queue, whereas in a lazily-evaluated context,
        it might mean that the returned array is a symbol bound to
        the data in *array*.

        The returned array may not be used with other contexts while thawed.
        """

    def freeze_thaw(
            self, array: ArrayOrContainerOrScalarT
            ) -> ArrayOrContainerOrScalarT:
        r"""Evaluate an input array or container to "frozen" data return a new
        "thawed" array or container representing the evaluation result that is
        ready for use. This is a shortcut for calling :meth:`freeze` and
        :meth:`thaw`.

        This method can be useful in array contexts backed by, e.g.
        :mod:`pytato`, to force the evaluation of a built-up array expression
        (and thereby avoid reevaluations for expressions built on the array).
        """
        return self.thaw(self.freeze(array))

    @abstractmethod
    def tag(self,
            tags: ToTagSetConvertible,
            array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        """If the array type used by the array context is capable of capturing
        metadata, return a version of *array* with the *tags* applied. *array*
        itself is not modified. When working with array containers, the
        tags are applied to each leaf of the container.

        See :ref:`metadata` as well as application-specific metadata types.

        .. versionadded:: 2021.2
        """

    @abstractmethod
    def tag_axis(self,
                 iaxis: int, tags: ToTagSetConvertible,
                 array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        """If the array type used by the array context is capable of capturing
        metadata, return a version of *array* in which axis number *iaxis* has
        the *tags* applied. *array* itself is not modified. When working with
        array containers, the tags are applied to each leaf of the container.

        See :ref:`metadata` as well as application-specific metadata types.

        .. versionadded:: 2021.2
        """

    @memoize_method
    def _get_einsum_prg(self,
                        spec: str, arg_names: Tuple[str, ...],
                        tagged: ToTagSetConvertible) -> "loopy.TranslationUnit":
        import loopy as lp
        from loopy.version import MOST_RECENT_LANGUAGE_VERSION

        from .loopy import _DEFAULT_LOOPY_OPTIONS
        return lp.make_einsum(
            spec,
            arg_names,
            options=_DEFAULT_LOOPY_OPTIONS,
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            tags=tagged,
            default_order=lp.auto,
            default_offset=lp.auto,
        )

    # This lives here rather than in .np because the interface does not
    # agree with numpy's all that well. Why can't it, you ask?
    # Well, optimizing generic einsum for OpenCL/GPU execution
    # is actually difficult, even in eager mode, and so without added
    # metadata describing what's happening, transform_loopy_program
    # has a very difficult (hopeless?) job to do.
    #
    # Unfortunately, the existing metadata support (cf. .tag()) cannot
    # help with eager mode execution [1], because, by definition, when the
    # result is passed to .tag(), it is already computed.
    # That's why einsum's interface here needs to be cluttered with
    # metadata, and that's why it can't live under .np.
    # [1] https://github.com/inducer/meshmode/issues/177
    def einsum(self,
               spec: str, *args: Array,
               arg_names: Optional[Tuple[str, ...]] = None,
               tagged: ToTagSetConvertible = ()) -> Array:
        """Computes the result of Einstein summation following the
        convention in :func:`numpy.einsum`.

        :arg spec: a string denoting the subscripts for
            summation as a comma-separated list of subscript labels.
            This follows the usual :func:`numpy.einsum` convention.
            Note that the explicit indicator `->` for the precise output
            form is required.
        :arg args: a sequence of array-like operands, whose order matches
            the subscript labels provided by *spec*.
        :arg arg_names: an optional iterable of string types denoting
            the names of the *args*. If *None*, default names will be
            generated.
        :arg tagged: an optional sequence of :class:`pytools.tag.Tag`
            objects specifying the tags to be applied to the operation.

        :return: the output of the einsum :mod:`loopy` program
        """
        if arg_names is None:
            arg_names = tuple(f"arg{i}" for i in range(len(args)))

        prg = self._get_einsum_prg(spec, arg_names, tagged)
        out_ary = self.call_loopy(
            prg, **{arg_names[i]: arg for i, arg in enumerate(args)}
        )["out"]
        return self.tag(tagged, out_ary)

    @abstractmethod
    def clone(self: SelfType) -> SelfType:
        """If possible, return a version of *self* that is semantically
        equivalent (i.e. implements all array operations in the same way)
        but is a separate object. May return *self* if that is not possible.

        .. note::

            The main objective of this semi-documented method is to help
            flag errors more clearly when array contexts are mixed that
            should not be. For example, at the time of this writing,
            :class:`meshmode.meshmode.Discretization` objects have a private
            array context that is only to be used for setup-related tasks.
            By using :meth:`clone` to make this a separate array context,
            and by checking that arithmetic does not mix array contexts,
            it becomes easier to detect and flag if unfrozen data attached to a
            "setup-only" array context "leaks" into the application.
        """

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        """Compiles *f* for repeated use on this array context. *f* is expected
        to be a `pure function <https://en.wikipedia.org/wiki/Pure_function>`__
        performing an array computation.

        Control flow statements (``if``, ``while``) that might take different
        paths depending on the data lead to undefined behavior and are illegal.
        Any data-dependent control flow must be expressed via array functions,
        such as ``actx.np.where``.

        *f* may be called on placeholder data, to obtain a representation
        of the computation performed, or it may be called as part of the actual
        computation, on actual data. If *f* is called on placeholder data,
        it may be called only once (or a few times).

        :arg f: the function executing the computation.
        :return: a function with the same signature as *f*.
        """
        return f

    # FIXME: Think about making this a standalone function? Would make it easier to
    # pass arguments when used as a decorator, e.g.:
    #     @outline(actx, id=...)
    #     def func(...):
    # vs.
    #     outline = partial(actx.outline, id=...)
    #
    #     @outline
    #     def func(...):
    def outline(self,
                f: Callable[..., Any],
                *,
                id: Optional[Hashable] = None) -> Callable[..., Any]:
        """
        Returns a drop-in-replacement for *f*. The behavior of the returned
        callable is specific to the derived class.

        The reason for the existence of such a routine is mainly for
        arraycontexts that allow a lazy mode of execution. In such
        arraycontexts, the computations within *f* maybe staged to potentially
        enable additional compiler transformations. See
        :func:`pytato.function.trace_call` or :func:`jax.named_call` for
        examples.

        :arg f: the function executing the computation to be staged.
        :return: a function with the same signature as *f*.
        """
        return f

    # undocumented for now
    @property
    @abstractmethod
    def permits_inplace_modification(self) -> bool:
        """
        *True* if the arrays allow in-place modifications.
        """

    # undocumented for now
    @property
    @abstractmethod
    def supports_nonscalar_broadcasting(self) -> bool:
        """
        *True* if the arrays support non-scalar broadcasting.
        """

    # undocumented for now
    @property
    @abstractmethod
    def permits_advanced_indexing(self) -> bool:
        """
        *True* if the arrays support :mod:`numpy`'s advanced indexing semantics.
        """

# }}}


# {{{ tagging helpers

def tag_axes(
        actx: ArrayContext,
        dim_to_tags: Mapping[int, ToTagSetConvertible],
        ary: ArrayT) -> ArrayT:
    """
    Return a copy of *ary* with the axes in *dim_to_tags* tagged with their
    corresponding tags. Equivalent to repeated application of
    :meth:`ArrayContext.tag_axis`.
    """
    for iaxis, tags in dim_to_tags.items():
        ary = actx.tag_axis(iaxis, tags, ary)

    return ary

# }}}


class UntransformedCodeWarning(UserWarning):
    pass

# vim: foldmethod=marker
