The Array Context Abstraction
=============================

.. automodule:: arraycontext

.. automodule:: arraycontext.context

Implementations of the Array Context Abstraction
================================================

Array context based on :mod:`pyopencl.array`
--------------------------------------------

.. automodule:: arraycontext.impl.pyopencl


Lazy/Deferred evaluation array context based on :mod:`pytato`
-------------------------------------------------------------

.. automodule:: arraycontext.impl.pytato


Array context :mod:`jax.numpy`
-------------------------------------------------------------

.. automodule:: arraycontext.impl.jax

.. _numpy-coverage:

:mod:`numpy` coverage
---------------------

This is a list of functionality implemented by :attr:`arraycontext.ArrayContext.np`.

.. note::

   Only functions and methods that have at least one implementation are listed.

.. include:: numpy_coverage.rst
