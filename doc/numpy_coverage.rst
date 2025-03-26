
.. raw:: html

    <style> .red {color:red} </style>
    <style> .green {color:green} </style>

.. role:: red
.. role:: green

Array creation routines
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1

    * - Function
      - :class:`~arraycontext.PyOpenCLArrayContext`
      - :class:`~arraycontext.EagerJAXArrayContext`
      - :class:`~arraycontext.PytatoPyOpenCLArrayContext`
      - :class:`~arraycontext.PytatoJAXArrayContext`
      - :class:`~arraycontext.NumpyArrayContext`
      - :class:`~arraycontext.CupyArrayContext`
    * - :func:`numpy.empty_like`
      - :green:`Yes`
      - :green:`Yes`
      - :red:`No`
      - :red:`No`
      - :red:`No`
      - :red:`No`
    * - :func:`numpy.ones_like`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.zeros_like`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.full_like`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :red:`No`
      - :red:`No`
    * - :func:`numpy.copy`
      - :green:`Yes`
      - :green:`Yes`
      - :red:`No`
      - :red:`No`
      - :red:`No`
      - :red:`No`

Array manipulation routines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1

    * - Function
      - :class:`~arraycontext.PyOpenCLArrayContext`
      - :class:`~arraycontext.EagerJAXArrayContext`
      - :class:`~arraycontext.PytatoPyOpenCLArrayContext`
      - :class:`~arraycontext.PytatoJAXArrayContext`
      - :class:`~arraycontext.NumpyArrayContext`
      - :class:`~arraycontext.CupyArrayContext`
    * - :func:`numpy.reshape`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.ravel`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.transpose`
      - :red:`No`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.broadcast_to`
      - :red:`No`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.concatenate`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.stack`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`

Linear algebra
~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1

    * - Function
      - :class:`~arraycontext.PyOpenCLArrayContext`
      - :class:`~arraycontext.EagerJAXArrayContext`
      - :class:`~arraycontext.PytatoPyOpenCLArrayContext`
      - :class:`~arraycontext.PytatoJAXArrayContext`
      - :class:`~arraycontext.NumpyArrayContext`
      - :class:`~arraycontext.CupyArrayContext`
    * - :func:`numpy.vdot`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`

Logic Functions
~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1

    * - Function
      - :class:`~arraycontext.PyOpenCLArrayContext`
      - :class:`~arraycontext.EagerJAXArrayContext`
      - :class:`~arraycontext.PytatoPyOpenCLArrayContext`
      - :class:`~arraycontext.PytatoJAXArrayContext`
      - :class:`~arraycontext.NumpyArrayContext`
      - :class:`~arraycontext.CupyArrayContext`
    * - :func:`numpy.all`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.any`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.greater`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.greater_equal`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.less`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.less_equal`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.equal`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.not_equal`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`

Mathematical functions
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1

    * - Function
      - :class:`~arraycontext.PyOpenCLArrayContext`
      - :class:`~arraycontext.EagerJAXArrayContext`
      - :class:`~arraycontext.PytatoPyOpenCLArrayContext`
      - :class:`~arraycontext.PytatoJAXArrayContext`
      - :class:`~arraycontext.NumpyArrayContext`
      - :class:`~arraycontext.CupyArrayContext`
    * - :data:`numpy.sin`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.cos`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.tan`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.arcsin`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.arccos`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.arctan`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.arctan2`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.sinh`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.cosh`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.tanh`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.floor`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.ceil`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.sum`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.exp`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.log`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.log10`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.real`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.imag`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.conjugate`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.maximum`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.amax`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :red:`No`
      - :red:`No`
    * - :data:`numpy.minimum`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :func:`numpy.amin`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :red:`No`
      - :red:`No`
    * - :data:`numpy.sqrt`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.absolute`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
    * - :data:`numpy.fabs`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
      - :green:`Yes`
