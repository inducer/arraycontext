Welcome to :mod:`arraycontext`'s documentation!
===============================================

GPU arrays? Deferred-evaluation arrays? Just plain ``numpy`` arrays? You'd like your
code to work with all of them? No problem! Comes with pre-made array context
implementations for:

- :mod:`numpy`
- :mod:`pyopencl`
- :mod:`jax.numpy`
- :mod:`pytato` (for lazy/deferred evaluation)
- Debugging
- Profiling

:mod:`arraycontext` started life as an array abstraction for use with the
:mod:`meshmode` unstrucuted discretization package.

Design Guidelines
-----------------

Here are some of the guidelines we aim to follow in :mod:`arraycontext`. There
exist numerous other, related efforts, such as the `Python array API standard
<https://data-apis.org/array-api/latest/purpose_and_scope.html>`__. These
points may aid in clarifying and differentiating our objectives.

- The array context is about exposing the common subset of operations
  available in immutable and mutable arrays. As a result, the interface
  does *not* seek to support interfaces that provide, enable, or are typically
  used only with in-place mutation.

  For example: The equivalents of :func:`numpy.empty` were deprecated
  and will eventually be removed.

- Each array context offers a specific subset of of :mod:`numpy` under
  :attr:`arraycontext.ArrayContext.np`. Functions under this namespace
  must be unconditionally :mod:`numpy`-compatible, that is, they may not
  offer an interface beyond what numpy offers. Functions that are
  incompatible, for example by supporting tag metadata
  (cf. :meth:`arraycontext.ArrayContext.einsum`) should live under the
  :class:`~arraycontext.ArrayContext` directly.

- Similarly, we strive to minimize redundancy between attributes of
  :class:`~arraycontext.ArrayContext` and :attr:`arraycontext.ArrayContext.np`.

  For example: ``ArrayContext.empty_like`` was deprecated.

- Array containers are data structures that may contain arrays.
  See :mod:`arraycontext.container`. We strive to support these, where sensible,
  in :class:`~arraycontext.ArrayContext` and :attr:`arraycontext.ArrayContext.np`.

Contents
--------

.. toctree::
    array_context
    implementations
    container
    other
    misc
    🚀 Github <https://github.com/inducer/arraycontext>
    💾 Download Releases <https://pypi.org/project/arraycontext>

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
