arraycontext: Choose your favorite ``numpy``-workalike
======================================================

.. image:: https://gitlab.tiker.net/inducer/arraycontext/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/arraycontext/commits/main
.. image:: https://github.com/inducer/arraycontext/workflows/CI/badge.svg
    :alt: Github Build Status
    :target: https://github.com/inducer/arraycontext/actions?query=branch%3Amain+workflow%3ACI
.. image:: https://badge.fury.io/py/arraycontext.png
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/arraycontext/

GPU arrays? Deferred-evaluation arrays? Just plain ``numpy`` arrays? You'd like your
code to work with all of them? No problem! Comes with pre-made array context
implementations for:

- numpy
- `PyOpenCL <https://documen.tician.de/pyopencl/array.html>`__
- `Pytato <https://documen.tician.de/pytato>`__ (for lazy/deferred evaluation)
- Debugging
- Profiling

``arraycontext`` started life as an array abstraction for use with the
`meshmode <https://documen.tician.de/meshmode/>`__ unstrucuted discretization
package.

Distributed under the MIT license.

Links
-----

* `Source code on Github <https://github.com/inducer/arraycontext>`_
* `Documentation <https://documen.tician.de/arraycontext>`_
