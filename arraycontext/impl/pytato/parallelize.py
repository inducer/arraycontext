# pyright: reportAny=warning

from __future__ import annotations


__copyright__ = """
Copyright (C) 2022-23 Kaushik Kulkarni
Copyright (C) 2022-26 University of Illinois Board of Trustees
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
import contextlib
from dataclasses import dataclass
from functools import reduce
from itertools import pairwise
from typing import TYPE_CHECKING, cast

from typing_extensions import override

import loopy as lp
from loopy.match import Matchable, MatchExpressionBase
from loopy.symbolic import WalkMapper
from loopy.translation_unit import CallablesTable, for_each_kernel


if TYPE_CHECKING:
    from collections.abc import Mapping

import logging


logger = logging.getLogger(__name__)


__doc__ = """
.. autofunction:: parallelize_disjoint_loop_sets
.. autofunction:: alias_global_temporaries
"""


# {{{ disjoint loop sets

@dataclass(frozen=True, eq=True)
class LoopSet:
    inames: frozenset[str]
    insns_in_loop_set: frozenset[str]


def get_disjoint_loop_sets(kernel: lp.LoopKernel) -> frozenset[LoopSet]:
    """
    Returns information about the disjoint loop sets in *kernel*.
    """
    disjoint_inames_and_insns: list[tuple[set[str], set[str]]] = []
    iname_to_associated_inames_and_insns: dict[str, tuple[set[str], set[str]]] = {}
    for insn in kernel.instructions:
        inames = insn.within_inames | insn.reduction_inames()
        associated_inames_and_insns: tuple[set[str], set[str]] | None = None
        for iname in inames:
            with contextlib.suppress(KeyError):
                associated_inames_and_insns = \
                    iname_to_associated_inames_and_insns[iname]
        if associated_inames_and_insns is not None:
            associated_inames, associated_insns = associated_inames_and_insns
            associated_inames.update(inames)
            associated_insns.add(insn.id)
        else:
            associated_inames_and_insns = (set(inames), {insn.id})
            disjoint_inames_and_insns.append(associated_inames_and_insns)
        for iname in inames:
            iname_to_associated_inames_and_insns[iname] = associated_inames_and_insns

    return frozenset({
        LoopSet(
            frozenset(associated_inames),
            frozenset(associated_insns))
        for associated_inames, associated_insns in disjoint_inames_and_insns})


def get_loop_set_dep_graph(
        kernel: lp.LoopKernel,
        loop_sets: frozenset[LoopSet]) -> dict[LoopSet, set[LoopSet]]:
    insn_id_to_loop_set = {
        insn_id: loop_set
        for loop_set in loop_sets
        for insn_id in loop_set.insns_in_loop_set}

    loop_set_dep_graph: dict[LoopSet, set[LoopSet]] = {
        insn_id_to_loop_set[insn.id]: set()
        for insn in kernel.instructions
    }

    for insn in kernel.instructions:
        insn_loop_set = insn_id_to_loop_set[insn.id]
        for dep_id in insn.depends_on:
            dep_loop_set = insn_id_to_loop_set[dep_id]
            if insn_loop_set != dep_loop_set:
                loop_set_dep_graph[dep_loop_set].add(insn_loop_set)

    return loop_set_dep_graph

# }}}


# {{{ split_iteration_domain_across_work_items

def get_iname_approx_length(kernel: lp.LoopKernel, iname: str) -> float | int:
    from loopy.isl_helpers import static_max_of_pw_aff
    max_domain_size = static_max_of_pw_aff(
        kernel.get_iname_bounds(iname).size,
        constants_only=False).to_pw_aff().max_val()
    if max_domain_size.is_infty():
        import math
        return math.inf
    else:
        return max_domain_size.to_python()


class OuterReductionNestCollector(WalkMapper[[]]):
    def __init__(self, outer_inames: frozenset[str]) -> None:
        super().__init__()
        self.outer_inames: frozenset[str] = outer_inames
        # Since we're only looking for the reductions that are on the outside, we can
        # use a list instead of a full graph
        self.outer_redn_nest: list[frozenset[str]] = []

    @override
    def map_reduction(self, expr: lp.Reduction) -> None:
        if not self.visit(expr):
            return

        outer_redn_inames = frozenset(expr.inames) & self.outer_inames

        if outer_redn_inames:
            self.outer_redn_nest.append(outer_redn_inames)

        self.rec(expr.expr)


def get_outer_iname_pos_from_loop_set(
        kernel: lp.LoopKernel, loop_set: LoopSet, outer_inames: frozenset[str]
        ) -> Mapping[str, int]:
    if not outer_inames:
        return {}

    import pymbolic.primitives as prim

    iname_orders: set[tuple[frozenset[str], ...]] = set()

    for insn_id in loop_set.insns_in_loop_set:
        insn = kernel.id_to_insn[insn_id]
        if isinstance(insn, lp.Assignment):
            insn_iname_order: list[frozenset[str]] = []
            if isinstance(insn.assignee, prim.Subscript):
                insn_iname_order.extend(
                    frozenset({idx.name})
                    for idx in insn.assignee.index_tuple
                    if (
                        isinstance(idx, prim.Variable)
                        and idx.name in outer_inames))
            ornc = OuterReductionNestCollector(outer_inames)
            ornc(insn.expression)
            insn_iname_order.extend(ornc.outer_redn_nest)
            if insn_iname_order:
                iname_orders.add(tuple(insn_iname_order))
        elif isinstance(insn, lp.CallInstruction):
            # must be a callable kernel, don't touch.
            pass
        elif isinstance(insn, (lp.BarrierInstruction, lp.NoOpInstruction)):
            pass
        else:
            raise NotImplementedError(type(insn))

    iname_order = None

    if iname_orders:
        # Merge the per-assignee partial orders into a single total order
        from pytools.graph import CycleError, compute_topological_order

        successors: dict[str, set[str]] = {iname: set() for iname in outer_inames}
        for order in iname_orders:
            for earlier, later in pairwise(order):
                for earlier_iname in earlier:
                    for later_iname in later:
                        successors[earlier_iname].add(later_iname)

        with contextlib.suppress(CycleError):
            # key= for determinism
            iname_order = compute_topological_order(successors, key=lambda x: x)

    if not iname_order:
        # No consistent merge of the per-assignee orderings exists; fall
        # back to a deterministic order based on iname names
        iname_order = sorted(outer_inames)

    return {iname: i
            for i, iname in enumerate(iname_order)}


def split_loop_set_across_work_items(
        kernel: lp.LoopKernel,
        callables: CallablesTable,
        loop_set: LoopSet,
        iname_to_approx_length: Mapping[str, float | int],
        max_device_compute_units: int,
) -> lp.LoopKernel:
    # Could possibly do something fancier that also includes the individual inner
    # loops in the loop set, but for now just looking at the inames shared between
    # all instructions in the set

    outer_non_redn_inames = loop_set.inames
    for insn_id in loop_set.insns_in_loop_set:
        outer_non_redn_inames &= kernel.id_to_insn[insn_id].within_inames

    outer_redn_inames = loop_set.inames
    for insn_id in loop_set.insns_in_loop_set:
        outer_redn_inames &= kernel.id_to_insn[insn_id].reduction_inames()

    outer_iname_pos: Mapping[str, int]
    all_outer_inames = outer_non_redn_inames | outer_redn_inames
    if all_outer_inames:
        outer_iname_pos = get_outer_iname_pos_from_loop_set(
            kernel, loop_set, all_outer_inames)
    else:
        outer_iname_pos = {}

    # Prioritize the non-reduction loop with largest loop count. In case of ties,
    # look at the iname position in the assignee and pick the iname indexing over
    # leading axis for the work-group hardware iname
    inames_to_parallelize = sorted(
        outer_non_redn_inames,
        key=lambda iname: (
            iname_to_approx_length[iname],
            -outer_iname_pos[iname]))

    # Add the largest reduction loop if we don't already have 2 non-reduction loops
    # to parallelize over
    if len(inames_to_parallelize) < 2 and outer_redn_inames:
        inames_to_parallelize.insert(0,
            max(
                outer_redn_inames,
                key=lambda iname: (
                    iname_to_approx_length[iname],
                    -outer_iname_pos[iname])))

    vng = kernel.get_var_name_generator()

    if len(inames_to_parallelize) == 0:
        pass
    elif len(inames_to_parallelize) == 1:
        iname, = inames_to_parallelize
        if iname in outer_non_redn_inames:
            # TODO: Compare performance with the commented-out version

            ngroups = max_device_compute_units * 4  # '4' to overfill the device
            local_zero_size = 64

            chunk_iname = vng(f"{iname}_chunk")
            inner_iname = vng(f"{iname}_inner")
            kernel = lp.split_iname(
                kernel, iname, ngroups * local_zero_size,
                outer_iname=chunk_iname, inner_iname=inner_iname)
            group_iname = vng(f"{iname}_group")
            local_zero_iname = vng(f"{iname}_local_zero")
            kernel = lp.split_iname(
                kernel, inner_iname, local_zero_size,
                outer_iname=group_iname, inner_iname=local_zero_iname,
                outer_tag="g.0", inner_tag="l.0")

            # ngroups = max_device_compute_units * 4  # '4' to overfill the device
            # local_one_size = 4
            # local_zero_size = 16

            # kernel = lp.split_iname(
            #     kernel, iname, ngroups * local_zero_size * local_one_size)
            # kernel = lp.split_iname(
            #     kernel, f"{iname}_inner", local_zero_size, inner_tag="l.0")
            # kernel = lp.split_iname(
            #     kernel, f"{iname}_inner_outer", local_one_size, inner_tag="l.1",
            #     outer_tag="g.0")

        else:
            from loopy.match import Id
            from loopy.transform.data import reduction_arg_to_subst_rule
            from loopy.transform.precompute import precompute_for_single_kernel

            # TODO: Make size-aware
            ngroups = max_device_compute_units
            # FIXME: local_one_size > 1 not working at the moment
            # local_one_size = 2
            local_one_size = 1
            local_zero_size = 32

            chunk_iname = vng(f"{iname}_chunk")
            inner_iname = vng(f"{iname}_inner")
            kernel = lp.split_iname(
                kernel, iname, ngroups * local_one_size * local_zero_size,
                outer_iname=chunk_iname, inner_iname=inner_iname)

            group_iname = vng(f"{iname}_group")
            local_zero_iname = vng(f"{iname}_local_zero")
            # This group_iname is for the final (serial) stage of the reduction,
            # so it doesn't get tagged with g.0
            kernel = lp.split_iname(
                kernel, inner_iname, local_one_size * local_zero_size,
                outer_iname=group_iname, inner_iname=local_zero_iname,
                inner_tag="l.0")
            kernel = lp.split_reduction_outward(kernel, group_iname)
            kernel = lp.split_reduction_outward(kernel, local_zero_iname)

            # group_iname = vng(f"{iname}_group")
            # local_iname = vng(f"{iname}_local")
            # # This group_iname is for the final (serial) stage of the reduction,
            # # so it doesn't get tagged with g.0
            # kernel = lp.split_iname(
            #     kernel, inner_iname, local_one_size * local_zero_size,
            #     outer_iname=group_iname, inner_iname=local_iname)
            # kernel = lp.split_reduction_outward(kernel, group_iname)

            # local_one_iname = vng(f"{iname}_local_one")
            # local_zero_iname = vng(f"{iname}_local_zero")
            # kernel = lp.split_iname(
            #     kernel, local_iname, local_zero_size,
            #     outer_iname=local_one_iname, inner_iname=local_zero_iname,
            #     outer_tag="l.1", inner_tag="l.0")
            # kernel = lp.split_reduction_outward(kernel, local_one_iname)
            # kernel = lp.split_reduction_outward(kernel, local_zero_iname)

            insn_ids = sorted(loop_set.insns_in_loop_set)

            iprcmpt_redn_group = vng(f"iprcmpt_{group_iname}")

            compute_insns: list[str] = []
            for insn_id in insn_ids:
                subst_rule_name = vng(f"redn_subst_{iname}_{insn_id}")
                kernel = reduction_arg_to_subst_rule(
                    kernel, group_iname,
                    subst_rule_name=subst_rule_name,
                    insn_match=Id(insn_id))

                temp_name = vng(f"redn_temp_{iname}_{insn_id}")
                compute_insn_id = vng(f"redn_compute_{iname}_{insn_id}")
                kernel = precompute_for_single_kernel(
                    kernel, callables, subst_rule_name, group_iname,
                    temporary_name=temp_name,
                    temporary_address_space=lp.AddressSpace.GLOBAL,
                    precompute_inames=[iprcmpt_redn_group],
                    default_tag="g.0",
                    # Don't add a barrier here, for two reasons:
                    # 1) this will create a separate barrier for each temporary
                    #    (only one is needed, because instructions inside a
                    #    reduction-only outer loop can't depend on each other)
                    # 2) add_gbarrier_between_disjoint_loop_sets only works if
                    #    barriers have not yet been added.
                    add_barrier_for_global_temporary=False,
                    compute_insn_id=compute_insn_id)

                compute_insns.append(compute_insn_id)

    else:
        if inames_to_parallelize[-2] in outer_non_redn_inames:
            bigger_iname = inames_to_parallelize[-1]
            smaller_iname = inames_to_parallelize[-2]

            # TODO: Make size-aware
            ngroups = max_device_compute_units * 4  # '4' to overfill the device
            local_one_size = 4
            local_zero_size = 16

            bigger_chunk_iname = vng(f"{bigger_iname}_chunk")
            bigger_inner_iname = vng(f"{bigger_iname}_inner")
            kernel = lp.split_iname(
                kernel, bigger_iname, ngroups * local_one_size,
                outer_iname=bigger_chunk_iname, inner_iname=bigger_inner_iname)

            # TODO: Think about whether lp.join_inames could be used below

            group_iname = vng(f"{bigger_iname}_group")
            local_one_iname = vng(f"{bigger_iname}_local_one")
            kernel = lp.split_iname(
                kernel, bigger_inner_iname, local_one_size,
                outer_iname=group_iname, inner_iname=local_one_iname,
                outer_tag="g.0", inner_tag="l.1")

            smaller_chunk_iname = vng(f"{smaller_iname}_chunk")
            local_zero_iname = vng(f"{smaller_iname}_local_zero")
            kernel = lp.split_iname(
                kernel, smaller_iname, local_zero_size,
                outer_iname=smaller_chunk_iname, inner_iname=local_zero_iname,
                inner_tag="l.0")

        else:
            non_redn_iname = inames_to_parallelize[-1]
            redn_iname = inames_to_parallelize[-2]

            # TODO: Make size-aware
            ngroups = max_device_compute_units * 4  # '4' to overfill the device
            local_zero_size = 32

            non_redn_length = iname_to_approx_length[non_redn_iname]
            redn_length = iname_to_approx_length[redn_iname]

            if non_redn_length/ngroups > redn_length:
                chunk_iname = vng(f"{non_redn_iname}_chunk")
                inner_iname = vng(f"{non_redn_iname}_inner")
                kernel = lp.split_iname(
                    kernel, non_redn_iname, ngroups * local_zero_size,
                    outer_iname=chunk_iname, inner_iname=inner_iname)

                group_iname = vng(f"{non_redn_iname}_group")
                local_zero_iname = vng(f"{non_redn_iname}_local_zero")
                kernel = lp.split_iname(
                    kernel, inner_iname, local_zero_size,
                    outer_iname=group_iname, inner_iname=local_zero_iname,
                    outer_tag="g.0", inner_tag="l.0")
            else:
                non_redn_chunk_iname = vng(f"{non_redn_iname}_chunk")
                group_iname = vng(f"{non_redn_iname}_group")
                kernel = lp.split_iname(
                    kernel, non_redn_iname, ngroups,
                    outer_iname=non_redn_chunk_iname, inner_iname=group_iname,
                    inner_tag="g.0")

                redn_chunk_iname = vng(f"{redn_iname}_chunk")
                local_zero_iname = vng(f"{redn_iname}_local_zero")
                kernel = lp.split_iname(
                    kernel, redn_iname, local_zero_size,
                    outer_iname=redn_chunk_iname, inner_iname=local_zero_iname,
                    inner_tag="l.0")
                kernel = lp.split_reduction_outward(kernel, local_zero_iname)

    return kernel


@for_each_kernel
def split_iteration_domain_across_work_items_for_single_kernel(
        kernel: lp.LoopKernel,
        callables: CallablesTable,
        max_device_compute_units: int, *,
        single_launch_config: bool = False) -> lp.LoopKernel:
    if single_launch_config:
        raise NotImplementedError("single_launch_config==True isn't implemented yet.")

    iname_to_approx_length = {
        iname: get_iname_approx_length(kernel, iname)
        for iname in kernel.all_inames()}

    loop_sets = get_disjoint_loop_sets(kernel)

    for loop_set in loop_sets:
        kernel = split_loop_set_across_work_items(
            kernel, callables, loop_set, iname_to_approx_length,
            max_device_compute_units)

    return kernel


def split_iteration_domain_across_work_items(
        t_unit: lp.TranslationUnit,
        max_device_compute_units: int, *,
        single_launch_config: bool = False) -> lp.TranslationUnit:
    """
    Tag inames in *t_unit* with work-group/work-item axes so that each disjoint
    loop set is parallelized across the device. Loops are split based on their
    approximate length and *max_device_compute_units*.
    """
    # Need to pass callables table down into per-kernel function due to
    # precompute_for_single_kernel call
    return split_iteration_domain_across_work_items_for_single_kernel(
        t_unit, t_unit.callables_table,
        max_device_compute_units=max_device_compute_units,
        single_launch_config=single_launch_config)

# }}}


# {{{ add_gbarrier_between_disjoint_loop_sets

def assign_loop_sets_to_call_kernels(
        kernel: lp.LoopKernel,
        loop_sets: frozenset[LoopSet], *,
        single_launch_config: bool = False) -> tuple[frozenset[LoopSet], ...]:
    """
    Assemble the loop sets into a sequence of call kernels to launch.

    This version creates a separate kernel for each loop set.
    """
    from pytools.graph import compute_topological_order

    loop_set_dep_graph = get_loop_set_dep_graph(kernel, loop_sets)

    toposorted_loop_sets: list[LoopSet] = compute_topological_order(
        loop_set_dep_graph,
        # Break ties between ready loop sets using the lexicographically smallest
        # instruction ID in each set. Loop sets are disjoint by construction, so these
        # mins are unique across sets
        key=lambda ls: min(ls.insns_in_loop_set))

    if single_launch_config:
        # Assign loop sets to call kernels based on their dependencies

        loop_set_to_call_kernel = dict.fromkeys(toposorted_loop_sets, 0)
        for loop_set in toposorted_loop_sets:
            for succ in loop_set_dep_graph[loop_set]:
                loop_set_to_call_kernel[succ] = max(
                    loop_set_to_call_kernel[succ],
                    loop_set_to_call_kernel[loop_set] + 1)

        n_call_kernels = max(loop_set_to_call_kernel.values()) + 1
        call_kernels: list[set[LoopSet]] = [set() for _ in range(n_call_kernels)]
        for loop_set, iknl in loop_set_to_call_kernel.items():
            call_kernels[iknl].add(loop_set)

        return tuple(frozenset(call_kernel) for call_kernel in call_kernels)

    else:
        # Make a separate call kernel for each loop set
        return tuple(frozenset([loop_set]) for loop_set in toposorted_loop_sets)


@dataclass(frozen=True)
class InsnIds(MatchExpressionBase):
    insn_ids_to_match: frozenset[str]

    @override
    def __call__(self, kernel: lp.LoopKernel, matchable: Matchable):
        return matchable.id in self.insn_ids_to_match


def add_gbarrier_between_disjoint_loop_sets(
        t_unit: lp.TranslationUnit, *,
        single_launch_config: bool = False) -> lp.TranslationUnit:
    """
    Returns a copy of *t_unit* with barriers added between dependent disjoint loop
    sets.

    .. warning::

        This routine assumes that the entrypoint in *t_unit* does not yet contain
        any barriers.
    """
    kernel = t_unit.default_entrypoint
    ing = kernel.get_instruction_id_generator()

    # Make sure there aren't any pre-existing barriers, otherwise this procedure may
    # add duplicates
    assert len([
        insn for insn in kernel.instructions
        if isinstance(insn, lp.BarrierInstruction)
        and insn.synchronization_kind == "global"]) == 0

    loop_sets = get_disjoint_loop_sets(kernel)
    call_kernels = assign_loop_sets_to_call_kernels(
        kernel, loop_sets, single_launch_config=single_launch_config)

    call_kernel_insn_ids: tuple[frozenset[str], ...] = tuple(
        reduce(
            lambda a, b: a | b,
            (loop_set.insns_in_loop_set for loop_set in call_kernel),
            cast("frozenset[str]", frozenset()))
        for call_kernel in call_kernels)

    gbarrier_ids: list[str] = []

    for ibarrier, (insns_before, insns_after) in enumerate(
            pairwise(call_kernel_insn_ids)):
        id_based_on = ing(f"_actx_gbarrier_{ibarrier}")
        kernel = lp.add_barrier(
            kernel,
            insn_before=InsnIds(insns_before),
            insn_after=InsnIds(insns_after),
            id_based_on=id_based_on,
            within_inames=frozenset())
        assert id_based_on in kernel.id_to_insn
        gbarrier_ids.append(id_based_on)

    from loopy.match import Id
    for pred_gbarrier, succ_gbarrier in pairwise(gbarrier_ids):
        kernel = lp.add_dependency(kernel, Id(succ_gbarrier), pred_gbarrier)

    return t_unit.with_kernel(kernel)

# }}}


# {{{ parallelize_disjoint_loop_sets

def parallelize_disjoint_loop_sets(
        t_unit: lp.TranslationUnit,
        max_device_compute_units: int) -> lp.TranslationUnit:
    """
    Parallelize *t_unit* by tagging the inames of each disjoint loop set with
    work-group and work-item axes and enforcing ordering between dependent
    loop sets.
    """
    t_unit = split_iteration_domain_across_work_items(
        t_unit, max_device_compute_units)
    t_unit = add_gbarrier_between_disjoint_loop_sets(t_unit)
    return t_unit

# }}}


# {{{ global temp var aliasing for disjoint live intervals

def get_call_kernel_insn_ids(kernel: lp.LoopKernel) -> tuple[frozenset[str], ...]:
    """
    Returns a sequence of sets of instruction IDs where each entry in the
    sequence corresponds to the instructions in a call-kernel to launch.
    """
    from pytools.graph import compute_topological_order

    global_barrier_ids = frozenset(
        insn.id for insn in kernel.instructions
        if isinstance(insn, lp.BarrierInstruction)
        and insn.synchronization_kind == "global")

    dep_graph: dict[str, set[str]] = {
        insn.id: set() for insn in kernel.instructions}
    for insn in kernel.instructions:
        for dep_id in insn.depends_on:
            dep_graph[dep_id].add(insn.id)

    # key= for determinism
    toposorted_insn_ids = compute_topological_order(dep_graph, key=lambda x: x)

    insn_id_to_call_kernel: dict[str, int] = {}
    for insn_id in toposorted_insn_ids:
        insn = kernel.id_to_insn[insn_id]
        icall_kernel = 0
        for dep_id in insn.depends_on:
            icall_kernel = max(
                icall_kernel,
                insn_id_to_call_kernel[dep_id]
                + (1 if dep_id in global_barrier_ids else 0))
        insn_id_to_call_kernel[insn_id] = icall_kernel

    n_call_kernels = len(global_barrier_ids) + 1
    call_kernel_insn_ids: list[set[str]] = [set() for _ in range(n_call_kernels)]
    for insn_id, icall_kernel in insn_id_to_call_kernel.items():
        if insn_id not in global_barrier_ids:
            call_kernel_insn_ids[icall_kernel].add(insn_id)

    return tuple(frozenset(insn_ids) for insn_ids in call_kernel_insn_ids)


def alias_global_temporaries(t_unit: lp.TranslationUnit) -> lp.TranslationUnit:
    """
    Returns a copy of *t_unit* with temporaries of that have disjoint live
    intervals using the same :attr:`loopy.TemporaryVariable.base_storage`.
    """
    from collections import defaultdict

    import loopy as lp
    from loopy.kernel.data import AddressSpace
    from pytools import UniqueNameGenerator

    t_unit = lp.infer_unknown_types(t_unit)

    # all loopy programs from pytato DAGs have exactly one entrypoint.
    kernel = t_unit.default_entrypoint

    temp_vars = frozenset(tv.name
                          for tv in kernel.temporary_variables.values()
                          if tv.address_space == AddressSpace.GLOBAL)

    call_kernel_insn_ids = get_call_kernel_insn_ids(kernel)
    expanded_kernel = lp.expand_subst(kernel)
    temp_to_live_interval_start: dict[str, int] = {}
    temp_to_live_interval_end: dict[str, int] = {}

    for icall_kernel, insn_ids in enumerate(call_kernel_insn_ids):
        for insn_id in insn_ids:
            for var in (expanded_kernel.id_to_insn[insn_id].dependency_names()
                        & temp_vars):
                if var not in temp_to_live_interval_start:
                    assert var not in temp_to_live_interval_end
                    temp_to_live_interval_start[var] = icall_kernel
                assert var in temp_to_live_interval_start
                temp_to_live_interval_end[var] = icall_kernel

    vng = UniqueNameGenerator()

    # {{{ get mappings from icall_kernel to temps that are just alive or dead

    icall_kernel_to_just_live_temp_vars: list[set[str]] = [
        set() for _ in call_kernel_insn_ids]
    icall_kernel_to_just_dead_temp_vars: list[set[str]] = [
        set() for _ in call_kernel_insn_ids]

    for tv_name, just_alive_idx in temp_to_live_interval_start.items():
        icall_kernel_to_just_live_temp_vars[just_alive_idx].add(tv_name)

    for tv_name, just_dead_idx in temp_to_live_interval_end.items():
        if just_dead_idx != (len(call_kernel_insn_ids) - 1):
            # we ignore the temporaries that died at the last kernel since we cannot
            # reclaim their memory
            icall_kernel_to_just_dead_temp_vars[just_dead_idx+1].add(tv_name)

    # }}}

    new_tvs: dict[str, lp.TemporaryVariable] = {}
    # a mapping from size in bytes to the available base storages from temp variables
    # that were dead.
    nbytes_to_available_base_storage: dict[int, set[str]] = defaultdict(set)

    for icall_kernel, _ in enumerate(call_kernel_insn_ids):
        just_dead_temps = icall_kernel_to_just_dead_temp_vars[icall_kernel]
        to_be_allocated_temps = icall_kernel_to_just_live_temp_vars[icall_kernel]

        # reclaim base storage from the dead temporaries
        for tv_name in sorted(just_dead_temps):
            tv = new_tvs[tv_name]
            assert tv.base_storage is not None
            assert isinstance(tv.nbytes, int)
            assert tv.base_storage not in nbytes_to_available_base_storage[tv.nbytes]
            nbytes_to_available_base_storage[tv.nbytes].add(tv.base_storage)

        # assign base storages to 'to_be_allocated_temps'
        for tv_name in sorted(to_be_allocated_temps):
            tv = kernel.temporary_variables[tv_name]
            assert tv.name not in new_tvs
            assert tv.base_storage is None
            assert isinstance(tv.nbytes, int)
            if nbytes_to_available_base_storage[tv.nbytes]:
                base_storage = sorted(nbytes_to_available_base_storage[tv.nbytes])[0]
                nbytes_to_available_base_storage[tv.nbytes].remove(base_storage)
            else:
                base_storage = vng("_actx_tmp_base")

            new_tvs[tv.name] = tv.copy(base_storage=base_storage)

    for name, tv in kernel.temporary_variables.items():
        if tv.address_space != AddressSpace.GLOBAL:
            new_tvs[name] = tv

    kernel = kernel.copy(temporary_variables=new_tvs)
    kernel = lp.allocate_temporaries_for_base_storage(kernel)

    def verify_is_int(x: object) -> int:
        assert isinstance(x, int)
        return x

    old_tmp_mem_requirement = sum(
        verify_is_int(tv.nbytes)
        for tv in kernel.temporary_variables.values())
    new_tmp_mem_requirement = sum(
        {tv.base_storage: verify_is_int(tv.nbytes)
         for tv in kernel.temporary_variables.values()}.values())
    logger.info(
        "[alias_global_temporaries]: Reduced memory requirement from "
        "%.1fMB to %.1fMB.",
        old_tmp_mem_requirement*1e-6, new_tmp_mem_requirement*1e-6)

    return t_unit.with_kernel(kernel)

# }}}


# vim: foldmethod=marker
