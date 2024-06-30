"""
.. autoclass:: SplitPytatoPyOpenCLArrayContext

"""

__copyright__ = """
Copyright (C) 2023 Kaushik Kulkarni
Copyright (C) 2023 Andreas Kloeckner
Copyright (C) 2022 Matthias Diener
Copyright (C) 2022 Matt Smith
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

import sys
from typing import TYPE_CHECKING

import loopy as lp

from arraycontext.impl.pytato import PytatoPyOpenCLArrayContext


if TYPE_CHECKING or getattr(sys, "_BUILDING_SPHINX_DOCS", False):
    import pytato


class SplitPytatoPyOpenCLArrayContext(PytatoPyOpenCLArrayContext):
    """
    .. note::

        Refer to :meth:`transform_dag` and :meth:`transform_loopy_program` for
        details on the transformation algorithm provided by this array context.

    .. warning::

        For expression graphs with large number of nodes high compile times are
        expected.
    """
    def transform_dag(self,
                      dag: "pytato.DictOfNamedArrays") -> "pytato.DictOfNamedArrays":
        r"""
        Returns a transformed version of *dag*, where the applied transform is:

        #. Materialize as per MPMS materialization heuristic.
        #. materialize every :class:`pytato.array.Einsum`\ 's inputs and outputs.
        """
        import pytato as pt

        # Step 1. Collapse equivalent nodes in DAG.
        # -----------------------------------------
        # type-ignore-reason: mypy is right pytato provides imprecise types.
        dag = pt.transform.deduplicate_data_wrappers(dag)  # type: ignore[assignment]

        # Step 2. Materialize reduction inputs/outputs.
        # ------------------------------------------
        from .utils import (
            get_inputs_and_outputs_of_einsum,
            get_inputs_and_outputs_of_reduction_nodes)

        reduction_inputs_outputs = frozenset.union(
            *get_inputs_and_outputs_of_einsum(dag),
            *get_inputs_and_outputs_of_reduction_nodes(dag)
        )

        def materialize_einsum(expr: pt.transform.ArrayOrNames
                               ) -> pt.transform.ArrayOrNames:
            if expr in reduction_inputs_outputs:
                if isinstance(expr, pt.InputArgumentBase):
                    return expr
                else:
                    return expr.tagged(pt.tags.ImplStored())
            else:
                return expr

        # type-ignore-reason: mypy is right pytato provides imprecise types.
        dag = pt.transform.map_and_copy(dag,  # type: ignore[assignment]
                                        materialize_einsum)

        # Step 3. MPMS materialize
        # ------------------------
        dag = pt.transform.materialize_with_mpms(dag)

        return dag

    def transform_loopy_program(self,
                                t_unit: lp.TranslationUnit) -> lp.TranslationUnit:
        r"""
        Returns a transformed version of *t_unit*, where the applied transform is:

        #. An execution grid size :math:`G` is selected based on *self*'s
           OpenCL-device.
        #. The iteration domain for each statement in the *t_unit* is divided to
           equally among the work-items in :math:`G`.
        #. Kernel boundaries are drawn between every statement in the instruction.
           Although one can relax this constraint by letting :mod:`loopy` compute
           where to insert the global barriers, but it is not guaranteed to be
           performance profitable since we do not attempt any further loop-fusion
           and/or array contraction.
        #. Once the kernel boundaries are inferred, :func:`alias_global_temporaries`
           is invoked to reduce the memory peak memory used by the transformed
           program.
        """
        # Step 1. Split the iteration across work-items
        # ---------------------------------------------
        from .utils import split_iteration_domain_across_work_items
        t_unit = split_iteration_domain_across_work_items(t_unit, self.queue.device)

        # Step 2. Add a global barrier between individual loop nests.
        # ------------------------------------------------------
        from .utils import add_gbarrier_between_disjoint_loop_nests
        t_unit = add_gbarrier_between_disjoint_loop_nests(t_unit)

        # Step 3. Transform reduce to scalar statements
        # ---------------------------------------------
        from .utils import parallelize_reduce_to_scalars
        t_unit = parallelize_reduce_to_scalars(t_unit, self.queue.device)

        # Step 4. Alias global temporaries with disjoint live intervals
        # -------------------------------------------------------------
        from .utils import alias_global_temporaries
        t_unit = alias_global_temporaries(t_unit)

        return t_unit

# vim: fdm=marker
