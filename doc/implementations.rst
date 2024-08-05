Implementations of the Array Context Abstraction
================================================

..
    When adding a new array context here, make sure to also add it to and run
    ```
        doc/make_numpy_coverage_table.py
    ```
    to update the coverage table below!

Array context based on :mod:`pyopencl.array`
--------------------------------------------

.. automodule:: arraycontext.impl.pyopencl


Lazy/Deferred evaluation array context based on :mod:`pytato`
-------------------------------------------------------------

.. automodule:: arraycontext.impl.pytato


Array context based on :mod:`jax.numpy`
---------------------------------------

.. automodule:: arraycontext.impl.jax

.. _numpy-coverage:

:mod:`numpy` coverage
---------------------

This is a list of functionality implemented by :attr:`arraycontext.ArrayContext.np`.

.. note::

   Only functions and methods that have at least one implementation are listed.

.. include:: numpy_coverage.rst

