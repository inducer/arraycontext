"""
.. autoclass:: BatchedEinsumPytatoPyOpenCLArrayContext """

__copyright__ = """
Copyright (C) 2023 Kaushik Kulkarni
Copyright (C) 2022 Andreas Kloeckner
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


import logging
import sys
from typing import TYPE_CHECKING, Any, Callable, Optional, Type
from warnings import warn

import numpy as np

import loopy as lp
from pytools import ProcessLogger
from pytools.tag import Tag

from arraycontext.impl.pytato import PytatoPyOpenCLArrayContext


logger = logging.getLogger(__name__)


if TYPE_CHECKING or getattr(sys, "_BUILDING_SPHINX_DOCS", False):
    import pyopencl as cl
    import pytato


class BatchedEinsumPytatoPyOpenCLArrayContext(PytatoPyOpenCLArrayContext):
    r"""
    .. attribute:: loop_fusion_axis_tag_t

        A subtype of :class:`pytato.tag.Tag` that are attached to the
        :class:`~pytato.array.Array`\ 's axes in an expression graph. Loops that
        iterate over axes tagged with instances of same such tag types will form the
        candidate loops for Kennedy's unweighted Loop Fusion algorithm.

    .. attribute:: fallback_to_no_fusion

        If *True*, during the compilation of an array expression graph for which
        loop fusion fails (see note) transformation routines from
        :class:`arraycontext.SplitPytatoPyOpenCLArrayContext` are invoked.

    .. attribute:: feinsum_db

        An instance of :class:`str` corresponding to the database of tuned batched
        einsums. If *None*, then a static transformation strategy is applied to the
        batched einsums kernels.

    .. attribute:: log_loopy_statistics

        If *True*, statistics of compiled :class:`loopy.TranslationUnit` will be
        logged. If enable, we log the FLOPS and global memory access footprint for
        each of the programs. If *False*, nothing is done.

    .. note::

        The conditions under which we fallback (or raise) are:

        #. There exists an array that is to be materialized but at least one of its
           axes is not tagged with tags of :attr:`loop_fusion_axis_tag_t`.
    """
    def __init__(
        self,
        queue: "cl.CommandQueue", allocator=None,
        *,
        loop_fusion_axis_tag_t: Type[Tag],
        fallback_to_no_fusion: bool = True,
        assume_all_indirection_maps_as_non_negative: bool = False,
        compile_trace_callback: Optional[Callable[[Any, str, Any], None]] = None,
        feinsum_db: Optional[str] = None,
        log_loopy_statistics: bool = False,
        fused_loop_name_prefix_getter: Optional[Callable[[Tag], str]] = None,
    ) -> None:
        super().__init__(queue,
                         allocator,
                         compile_trace_callback=compile_trace_callback)

        self.loop_fusion_axis_tag_t = loop_fusion_axis_tag_t
        self.fallback_to_no_fusion = fallback_to_no_fusion
        self.feinsum_db = feinsum_db
        self.assume_all_indirection_maps_as_non_negative = (
            assume_all_indirection_maps_as_non_negative)
        self.log_loopy_statistics = log_loopy_statistics
        if fused_loop_name_prefix_getter:
            self.fused_loop_name_prefix_getter = fused_loop_name_prefix_getter
        else:
            self.fused_loop_name_prefix_getter = lambda tag_t: "ifused"

    def transform_dag(self,
                      dag: "pytato.DictOfNamedArrays") -> "pytato.DictOfNamedArrays":
        import pytato as pt

        from .utils import (
            _make_passthrough_arg, get_indirection_maps,
            get_inputs_and_outputs_of_reduction_nodes)
        from arraycontext.impl.pytato.split_actx.utils import (
            get_inputs_and_outputs_of_einsum)

        # Step 1. Collapse equivalent nodes in DAG.
        # -----------------------------------------
        # type-ignore-reason: mypy is right pytato provides imprecise types.
        dag = pt.transform.deduplicate_data_wrappers(dag)  # type: ignore[assignment]

        # Step 2. Materialize einsum/reduction outputs.
        # ---------------------------------------------
        _, einsum_outputs = get_inputs_and_outputs_of_einsum(dag)
        _, reduction_outputs = get_inputs_and_outputs_of_reduction_nodes(dag)

        def materialize_all_einsums_or_reduces(expr):
            if (expr in einsum_outputs
                    or expr in reduction_outputs):
                return expr.tagged(pt.tags.ImplStored())
            else:
                return expr

        # type-ignore-reason: mypy is right pytato provides imprecise types.
        dag = pt.transform.map_and_copy(dag,  # type: ignore[assignment]
                                        materialize_all_einsums_or_reduces)

        # Step 3. Materialize with MPMS
        # -----------------------------
        dag = pt.transform.materialize_with_mpms(dag)

        # Step 4. Mark all indirection maps as non-negative
        # -------------------------------------------------
        if self.assume_all_indirection_maps_as_non_negative:
            indirection_maps = get_indirection_maps(dag)

            def tag_indices_as_non_negative(ary):
                if ary in indirection_maps:
                    return ary.tagged(pt.tags.AssumeNonNegative())
                else:
                    return ary

            # type-ignore-reason: mypy is right pytato provides imprecise types.
            dag = pt.transform.map_and_copy(dag,  # type: ignore[assignment]
                                            tag_indices_as_non_negative)

        # Step 5. Get rid of broadcasts in einsum expressions (helps feinsum)
        # -------------------------------------------------------------------
        dag = pt.rewrite_einsums_with_no_broadcasts(dag)

        # Step 6. Infer axis tags
        # -----------------------
        # type-ignore-reason: mypy is right pytato provides imprecise types.
        dag = pt.unify_axes_tags(dag)  # type: ignore[assignment]

        # Step 7. Make all pt.einsum/pt.reduction inputs as substitutions
        # ---------------------------------------------------------------
        def implement_einsum_reduction_inputs_as_substs(expr):
            from immutables import Map

            from pytato.target.loopy import ImplSubstitution
            if isinstance(expr, pt.Einsum):
                # make the arguments passthrough to make use of already stored
                # values.
                # pylint and 'attrs' have poor compatibility
                # pylint: disable=too-many-function-args,redundant-keyword-arg
                # pylint: disable=unexpected-keyword-arg
                return pt.Einsum(
                    expr.access_descriptors,
                    tuple(_make_passthrough_arg(arg, ImplSubstitution())
                          for arg in expr.args),
                    expr.redn_axis_to_redn_descr,
                    expr.index_to_access_descr,
                    tags=expr.tags,
                    axes=expr.axes,
                )
            elif isinstance(expr, pt.IndexLambda) and expr.var_to_reduction_descr:
                # make the arguments passthrough to make use of already stored
                # values.
                # pylint: disable=too-many-function-args,redundant-keyword-arg
                # pylint: disable=unexpected-keyword-arg
                return pt.IndexLambda(
                    expr.expr,
                    expr.shape,
                    expr.dtype,
                    Map({name: _make_passthrough_arg(bnd, ImplSubstitution())
                         for name, bnd in expr.bindings.items()}),
                    expr.var_to_reduction_descr,
                    tags=expr.tags,
                    axes=expr.axes,
                )
            else:
                return expr

        # type-ignore-reason: mypy is right pytato provides imprecise types.
        dag = pt.transform.map_and_copy(dag,  # type: ignore[assignment]
                                        implement_einsum_reduction_inputs_as_substs)

        return dag

    def transform_loopy_program(self,
                                t_unit: lp.TranslationUnit) -> lp.TranslationUnit:
        knl_name = t_unit.default_entrypoint.name

        logger.info(f"[{self.__class__}.transform_loopy_program]:"
                    f" Transforming kernel '{knl_name}' with"
                    f" {len(t_unit.default_entrypoint.instructions)} statements.")

        # Step 0. Fallback if cannot t_unit cannot be transformed
        # -------------------------------------------------------
        for iname in t_unit.default_entrypoint.all_inames():
            if not t_unit.default_entrypoint.iname_tags_of_type(
                    iname, self.loop_fusion_axis_tag_t):
                if self.fallback_to_no_fusion:
                    warn(f"[{knl_name}]: Falling back to a slower transformation"
                         " strategy as some loops are uninferred which mesh entity"
                         " they belong to.",
                         stacklevel=2)
                    from arraycontext.impl.pytato.split_actx import (
                        SplitPytatoPyOpenCLArrayContext)

                    # type-ignore-reason: mypy is right, we are passing incorrect
                    # types, but knowing the implementation of
                    # SplitPytatoPyOpenCLArrayContext this should be fine.
                    return SplitPytatoPyOpenCLArrayContext.transform_loopy_program(
                        self, t_unit)  # type: ignore[arg-type]
                else:
                    raise RuntimeError(f"Iname '{iname}' is not tagged with tags"
                                       f" of type '{self.loop_fusion_axis_tag_t}'"
                                       " => Not allowed since Kennedy's Loop fusion"
                                       " cannot be applied.")

        # Step 0.5. Make offsets as 0. (FIXME: move this to loopy knl invocation)
        # -----------------------------------------------------------------------
        knl = t_unit.default_entrypoint
        knl = knl.copy(args=[arg.copy(offset=0) for arg in knl.args])
        t_unit = t_unit.with_kernel(knl)
        del knl

        # Step 1. Fuse loops indexing over the same tag
        # ---------------------------------------------
        with ProcessLogger(logger, f"[{knl_name}] Loop Fusion"):
            from .utils import apply_kennedy_fusion_with_batched_einsum_extension
            t_unit = apply_kennedy_fusion_with_batched_einsum_extension(
                t_unit, self.loop_fusion_axis_tag_t,
                self.fused_loop_name_prefix_getter)

        # Step 2. Combine the domains of individual loop nests into individual
        # BasicSets
        # --------------------------------------------------------------------
        from .utils import combine_domains_of_perfect_loop_nests
        t_unit = combine_domains_of_perfect_loop_nests(t_unit)

        # Step 3. Remove dead temporaries
        # -------------------------------
        from .utils import remove_dead_temporaries
        t_unit = remove_dead_temporaries(t_unit)

        # Step 4. Contract arrays
        # -----------------------
        with ProcessLogger(logger, f"[{knl_name}] Array Contraction"):
            from .utils import contract_arrays
            t_unit = contract_arrays(t_unit)

        # Step 5. Collect statistics
        # --------------------------

        # {{{ compute stats

        if self.log_loopy_statistics:

            with ProcessLogger(logger, f"[{knl_name}] Count kernel metrics"):
                from loopy.kernel.array import ArrayBase
                from pytools import product
                knl = t_unit.default_entrypoint
                knl = knl.copy(
                    silenced_warnings=(knl.silenced_warnings
                                        + ["insn_count_subgroups_upper_bound",
                                            "summing_if_branches_ops"]))

                t_unit = t_unit.with_kernel(knl)
                del knl

                op_map = lp.get_op_map(t_unit, subgroup_size=32)

                c64_ops = {op_type: (op_map.filter_by(dtype=[np.complex64],
                                                      name=op_type,
                                                      kernel_name=knl_name)
                                      .eval_and_sum({}))
                            for op_type in ["add", "mul", "div"]}
                c128_ops = {op_type: (op_map.filter_by(dtype=[np.complex128],
                                                       name=op_type,
                                                       kernel_name=knl_name)
                                      .eval_and_sum({}))
                            for op_type in ["add", "mul", "div"]}
                f32_ops = ((op_map.filter_by(dtype=[np.float32],
                                             kernel_name=knl_name)
                            .eval_and_sum({}))
                           + (2 * c64_ops["add"]
                              + 6 * c64_ops["mul"]
                              + (6 + 3 + 2) * c64_ops["div"]))
                f64_ops = ((op_map.filter_by(dtype=[np.float64],
                                             kernel_name="_pt_kernel")
                            .eval_and_sum({}))
                           + (2 * c128_ops["add"]
                              + 6 * c128_ops["mul"]
                              + (6 + 3 + 2) * c128_ops["div"]))

                # {{{ footprint gathering

                nfootprint_bytes = 0

                for ary in knl.args:
                    if (isinstance(ary, ArrayBase)
                            and ary.address_space == lp.AddressSpace.GLOBAL):
                        nfootprint_bytes += (product(ary.shape)
                                            * ary.dtype.itemsize)

                for ary in knl.temporary_variables.values():
                    if ary.address_space == lp.AddressSpace.GLOBAL:
                        # global temps would be written once and read once
                        nfootprint_bytes += (2 * product(ary.shape)
                                            * ary.dtype.itemsize)

                # }}}

                if f32_ops:
                    logger.info(f"Single-prec. GFlOps: {f32_ops * 1e-9}")
                if f64_ops:
                    logger.info(f"Double-prec. GFlOps: {f64_ops * 1e-9}")
                logger.info(f"Footprint GBs: {nfootprint_bytes * 1e-9}")

        # }}}

        # Step 6. Draw kernel boundaries between batched einsum kernels
        # -------------------------------------------------------------
        from arraycontext.impl.pytato.split_actx.utils import (
            add_gbarrier_between_disjoint_loop_nests)

        t_unit = add_gbarrier_between_disjoint_loop_nests(t_unit)

        # Step 7. Alias global temporaries with disjoint live intervals
        # -------------------------------------------------------------
        from arraycontext.impl.pytato.split_actx.utils import (
            alias_global_temporaries)
        t_unit = alias_global_temporaries(t_unit)

        # Step 8. Macro-kernel optimizations
        # ----------------------------------
        if self.feinsum_db:
            from .utils import apply_feinsum_transformations
            t_unit = apply_feinsum_transformations(
                t_unit, self.feinsum_db, self.queue.device)
        else:
            from arraycontext.impl.pytato.split_actx.utils import (
                parallelize_reduce_to_scalars,
                split_iteration_domain_across_work_items)
            t_unit = split_iteration_domain_across_work_items(t_unit,
                                                              self.queue.device)
            t_unit = parallelize_reduce_to_scalars(t_unit, self.queue.device)

        return t_unit

# vim: fdm=marker
