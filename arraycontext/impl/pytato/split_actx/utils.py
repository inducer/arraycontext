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

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING, Any, Dict, FrozenSet, List, Mapping, Set, Tuple, Union)

import loopy as lp
import loopy.match as lp_match
import pytato as pt
from loopy.kernel.function_interface import InKernelCallable
from loopy.translation_unit import for_each_kernel
from pymbolic.mapper.optimize import optimize_mapper


if TYPE_CHECKING:
    import pyopencl


split_actx_utils_logger = logging.getLogger(__name__)


# {{{ EinsumInputOutputCollector

@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class EinsumInputOutputCollector(pt.transform.CachedWalkMapper):
    """
    .. note::

        We deliberately avoid using :class:`pytato.transform.CombineMapper` since
        the mapper's caching structure would still lead to recomputing
        the union of sets for the results of a revisited node.
    """
    def __init__(self) -> None:
        self.collected_outputs: Set[pt.Array] = set()
        self.collected_inputs: Set[pt.Array] = set()
        super().__init__()

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def get_cache_key(self,  # type: ignore[override]
                      expr: pt.transform.ArrayOrNames) -> int:
        return id(expr)

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def post_visit(self, expr: Any) -> None:  # type: ignore[override]
        if isinstance(expr, pt.Einsum):
            self.collected_outputs.add(expr)
            self.collected_inputs.update(expr.args)


def get_inputs_and_outputs_of_einsum(
        expr: pt.DictOfNamedArrays) -> Tuple[FrozenSet[pt.Array],
                                             FrozenSet[pt.Array]]:
    mapper = EinsumInputOutputCollector()
    mapper(expr)
    return frozenset(mapper.collected_inputs), frozenset(mapper.collected_outputs)

# }}}


# {{{ ReductionInputOutputCollector

class ReductionInputOutputCollector(pt.transform.CachedWalkMapper):
    """
    .. note::
        We deliberately avoid using :class:`pytato.transform.CombineMapper` since
        the mapper's caching structure would still lead to recomputing
        the union of sets for the results of a revisited node.
    """
    def __init__(self) -> None:
        self.collected_outputs: Set[pt.Array] = set()
        self.collected_inputs: Set[pt.Array] = set()
        super().__init__()

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def get_cache_key(self,  # type: ignore[override]
                      expr: pt.transform.ArrayOrNames) -> int:
        return id(expr)

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def post_visit(self, expr: Any) -> None:  # type: ignore[override]
        if isinstance(expr, pt.IndexLambda) and expr.var_to_reduction_descr:
            self.collected_outputs.add(expr)
            self.collected_inputs.update(expr.bindings.values())


def get_inputs_and_outputs_of_reduction_nodes(
        expr: pt.DictOfNamedArrays) -> Tuple[FrozenSet[pt.Array],
                                             FrozenSet[pt.Array]]:
    mapper = ReductionInputOutputCollector()
    mapper(expr)
    return frozenset(mapper.collected_inputs), frozenset(mapper.collected_outputs)

# }}}


# {{{ _LoopNest class definition

@dataclass(frozen=True, eq=True)
class _LoopNest:
    inames: FrozenSet[str]
    insns_in_loop_nest: FrozenSet[str]


def _is_a_perfect_loop_nest(kernel: lp.LoopKernel,
                            inames: FrozenSet[str]) -> bool:
    try:
        template_iname = next(iter(inames))
    except StopIteration:
        return True
    else:
        insn_ids_in_template_iname = kernel.iname_to_insns()[template_iname]
        return all(kernel.iname_to_insns()[iname] == insn_ids_in_template_iname
                   for iname in inames)


def _get_loop_nest(kernel: lp.LoopKernel, insn: lp.InstructionBase) -> _LoopNest:
    assert _is_a_perfect_loop_nest(kernel, insn.within_inames)
    if insn.within_inames:
        any_iname_in_nest, *other_inames = insn.within_inames
        return _LoopNest(insn.within_inames,
                         frozenset(kernel.iname_to_insns()[any_iname_in_nest]))
    else:
        if insn.reduction_inames():
            # TODO: Avoid O(N^2) complexity (typically there aren't long
            # kernels with "reduce-to-scalar" operations, but this might bite
            # in the future)
            insn_ids = frozenset({
                insn_.id
                for insn_ in kernel.instructions
                if insn_.reduction_inames() == insn.reduction_inames()})
            return _LoopNest(frozenset(), insn_ids)
        else:
            # we treat a loop nest with 0-depth in a special manner by putting
            # each such instruction into a separate loop nest.
            return _LoopNest(frozenset(), frozenset([insn.id]))

# }}}


# {{{ split_iteration_domain_across_work_items

def get_iname_length(kernel: lp.LoopKernel, iname: str) -> Union[float, int]:
    from loopy.isl_helpers import static_max_of_pw_aff
    max_domain_size = static_max_of_pw_aff(kernel.get_iname_bounds(iname).size,
                                           constants_only=False).max_val()
    if max_domain_size.is_infty():
        import math
        return math.inf
    else:
        return max_domain_size.to_python()


def _get_iname_pos_from_loop_nest(kernel, loop_nest: _LoopNest) -> Mapping[str, int]:
    import pymbolic.primitives as prim

    iname_orders: Set[Tuple[str, ...]] = set()

    for insn_id in loop_nest.insns_in_loop_nest:
        insn = kernel.id_to_insn[insn_id]
        if isinstance(insn, lp.Assignment):
            if isinstance(insn.assignee, prim.Subscript):
                iname_orders.add(tuple(idx.name
                                       for idx in insn.assignee.index_tuple))
        elif isinstance(insn, lp.CallInstruction):
            # must be a callable kernel, don't touch.
            pass
        elif isinstance(insn, (lp.BarrierInstruction, lp.NoOpInstruction)):
            pass
        else:
            raise NotImplementedError(type(insn))

    if len(iname_orders) != 1:
        raise RuntimeError("split_iteration_domain failed by receiving a"
                           " kernel not belonging to the expected grammar or"
                           " kernels.")

    iname_order, = iname_orders
    return {iname: i
            for i, iname in enumerate(iname_order)}


def _split_loop_nest_across_work_items(
        kernel: lp.LoopKernel,
        loop_nest: _LoopNest,
        iname_to_length: Mapping[str, Union[float, int]],
        cl_device: "pyopencl.Device",
) -> lp.LoopKernel:

    ngroups = cl_device.max_compute_units * 4  # '4' to overfill the device
    l_one_size = 4
    l_zero_size = 16

    if len(loop_nest.inames) == 0:
        pass
    elif len(loop_nest.inames) == 1:
        iname, = loop_nest.inames
        kernel = lp.split_iname(kernel, iname,
                                ngroups * l_zero_size * l_one_size)
        kernel = lp.split_iname(kernel, f"{iname}_inner",
                                l_zero_size, inner_tag="l.0")
        kernel = lp.split_iname(kernel, f"{iname}_inner_outer",
                                l_one_size, inner_tag="l.1",
                                outer_tag="g.0")
    else:
        iname_pos_in_assignee = _get_iname_pos_from_loop_nest(kernel, loop_nest)

        # Pick the loop with largest loop count. In case of ties, look at the
        # iname position in the assignee and pick the iname indexing over
        # leading axis for the work-group hardware iname.
        sorted_inames = sorted(loop_nest.inames,
                               key=lambda iname: (iname_to_length[iname],
                                                  -iname_pos_in_assignee[iname]))
        smaller_loop, bigger_loop = sorted_inames[-2], sorted_inames[-1]

        kernel = lp.split_iname(kernel, f"{bigger_loop}",
                                l_one_size * ngroups)
        kernel = lp.split_iname(kernel, f"{bigger_loop}_inner",
                                l_one_size, inner_tag="l.1", outer_tag="g.0")
        kernel = lp.split_iname(kernel, smaller_loop,
                                l_zero_size, inner_tag="l.0")

    return kernel


@for_each_kernel
def split_iteration_domain_across_work_items(
    kernel: lp.LoopKernel,
    cl_device: "pyopencl.Device",
) -> lp.LoopKernel:

    insn_id_to_loop_nest: Mapping[str, _LoopNest] = {
        insn.id: _get_loop_nest(kernel, insn)
        for insn in kernel.instructions
    }
    iname_to_length = {iname: get_iname_length(kernel, iname)
                       for iname in kernel.all_inames()}

    all_loop_nests = frozenset(insn_id_to_loop_nest.values())

    for loop_nest in all_loop_nests:
        kernel = _split_loop_nest_across_work_items(kernel,
                                                    loop_nest,
                                                    iname_to_length,
                                                    cl_device)

    return kernel

# }}}


# {{{ add_gbarrier_between_disjoint_loop_nests

@dataclass(frozen=True)
class InsnIds(lp_match.MatchExpressionBase):
    insn_ids_to_match: FrozenSet[str]

    def __call__(self, kernel: lp.LoopKernel, matchable: lp.InstructionBase):
        return matchable.id in self.insn_ids_to_match


def _get_call_kernel_insn_ids(kernel: lp.LoopKernel) -> Tuple[FrozenSet[str], ...]:
    """
    Returns a sequence of collection of instruction ids where each entry in the
    sequence corresponds to the instructions in a call-kernel to launch.

    In this heuristic we simply draw kernel boundaries such that instruction
    belonging to disjoint loop-nest pairs are executed in different call kernels.

    .. note::

        We require that every statement in *kernel* is nested within a perfect loop
        nest.
    """
    from pytools.graph import compute_topological_order

    loop_nest_dep_graph: Dict[_LoopNest, Set[_LoopNest]] = {
        _get_loop_nest(kernel, insn): set()
        for insn in kernel.instructions
    }

    for insn in kernel.instructions:
        insn_loop_nest = _get_loop_nest(kernel, insn)
        for dep_id in insn.depends_on:
            dep_loop_nest = _get_loop_nest(kernel, kernel.id_to_insn[dep_id])
            if insn_loop_nest != dep_loop_nest:
                loop_nest_dep_graph[dep_loop_nest].add(insn_loop_nest)

    # TODO: pass 'key' to compute_topological_order to ensure deterministic result
    toposorted_loop_nests: List[_LoopNest] = compute_topological_order(
        loop_nest_dep_graph)

    return tuple(loop_nest.insns_in_loop_nest for loop_nest in toposorted_loop_nests)


def add_gbarrier_between_disjoint_loop_nests(
        t_unit: lp.TranslationUnit) -> lp.TranslationUnit:
    kernel = t_unit.default_entrypoint
    ing = kernel.get_instruction_id_generator()

    call_kernel_insn_ids = _get_call_kernel_insn_ids(kernel)
    gbarrier_ids: List[str] = []

    for ibarrier, (insns_before, insns_after) in enumerate(
            zip(call_kernel_insn_ids[:-1], call_kernel_insn_ids[1:])):
        id_based_on = ing(f"_actx_gbarrier_{ibarrier}")
        kernel = lp.add_barrier(kernel,
                                insn_before=InsnIds(insns_before),
                                insn_after=InsnIds(insns_after),
                                id_based_on=id_based_on,
                                within_inames=frozenset())
        assert id_based_on in kernel.id_to_insn
        gbarrier_ids.append(id_based_on)

    for pred_gbarrier, succ_gbarrier in zip(gbarrier_ids[:-1], gbarrier_ids[1:]):
        kernel = lp.add_dependency(kernel, lp_match.Id(succ_gbarrier), pred_gbarrier)

    return t_unit.with_kernel(kernel)

# }}}


# {{{ parallelize_reduce_to_scalar_statements

def _split_reduce_to_scalar_across_work_items(
    kernel: lp.LoopKernel,
    callables: Mapping[str, InKernelCallable],
    insn_ids: FrozenSet[str],
    device: "pyopencl.Device",
) -> lp.LoopKernel:

    assert len({kernel.id_to_insn[insn_id].reduction_inames()
                for insn_id in insn_ids}) == 1
    redn_inames, = {kernel.id_to_insn[insn_id].reduction_inames()
                    for insn_id in insn_ids}
    if redn_inames:
        from loopy.transform.data import reduction_arg_to_subst_rule
        from loopy.transform.precompute import precompute_for_single_kernel

        iredn = max(redn_inames, key=lambda x: get_iname_length(kernel, x))
        serial_inames = redn_inames - frozenset([iredn])
        vng = kernel.get_var_name_generator()
        iredn_outer, iredn_inner = vng(f"{iredn}_outer"), vng(f"{iredn}_inner")
        iredn_inner_outer = vng(f"{iredn}_inner_outer")
        n_workgroups = device.max_compute_units
        wg_size = 32

        kernel = lp.split_iname(kernel, iredn, n_workgroups * wg_size,
                                inner_iname=iredn_inner, outer_iname=iredn_outer)
        kernel = lp.split_iname(kernel, iredn_inner, wg_size,
                                outer_iname=iredn_inner_outer, inner_tag="l.0")
        kernel = lp.split_reduction_outward(kernel, iredn_outer)
        kernel = lp.split_reduction_inward(kernel,
                                           serial_inames | {iredn_inner_outer})
        iprcmpt_redn_outer = vng(f"iprcmpt_{iredn_outer}")

        for insn_id in sorted(insn_ids):
            subst_rule_name = vng(f"redn_subst_{iredn}_{insn_id}")

            kernel = reduction_arg_to_subst_rule(kernel, iredn_outer,
                                                 subst_rule_name=subst_rule_name,
                                                 insn_match=lp_match.Id(insn_id))

            kernel = precompute_for_single_kernel(
                kernel, callables, subst_rule_name, iredn_outer,
                temporary_address_space=lp.AddressSpace.GLOBAL,
                precompute_inames=[iprcmpt_redn_outer],
                default_tag="g.0")

        return kernel
    else:
        return kernel


@for_each_kernel
def _parallelize_reduce_to_scalars_for_single_kernel(
    kernel: lp.LoopKernel,
    callables: Mapping[str, InKernelCallable],
    cl_device: "pyopencl.Device",
) -> lp.LoopKernel:

    # collect loop nests of instructions that assign to scalars in the array
    # program.
    insn_id_to_loop_nest: Mapping[str, _LoopNest] = {
        insn.id: _get_loop_nest(kernel, insn)
        for insn in kernel.instructions
        if (isinstance(insn, lp.Assignment)
            and not insn.within_inames)
    }

    all_loop_nests = frozenset(insn_id_to_loop_nest.values())

    for loop_nest in all_loop_nests:
        kernel = _split_reduce_to_scalar_across_work_items(
            kernel, callables, loop_nest.insns_in_loop_nest, cl_device)

    return kernel


def parallelize_reduce_to_scalars(t_unit: lp.TranslationUnit,
                                  cl_device: "pyopencl.Device"
                                  ) -> lp.TranslationUnit:
    return _parallelize_reduce_to_scalars_for_single_kernel(
        t_unit, t_unit.callables_table, cl_device)

# }}}


# {{{ global temp var aliasing for disjoint live intervals

def alias_global_temporaries(t_unit: lp.TranslationUnit) -> lp.TranslationUnit:
    """
    Returns a copy of *t_unit* with temporaries of that have disjoint live
    intervals using the same :attr:`loopy.TemporaryVariable.base_storage`.

    .. warning::

        This routine **assumes** that the entrypoint in *t_unit* global
        barriers inserted as per :func:`_get_call_kernel_insn_ids`.
    """

    from collections import defaultdict

    from loopy.kernel.data import AddressSpace
    from pytools import UniqueNameGenerator

    t_unit = lp.infer_unknown_types(t_unit)

    # all loopy programs from pytato DAGs have exactly one entrypoint.
    kernel = t_unit.default_entrypoint

    temp_vars = frozenset(tv.name
                          for tv in kernel.temporary_variables.values()
                          if tv.address_space == AddressSpace.GLOBAL)

    call_kernel_insn_ids = _get_call_kernel_insn_ids(kernel)
    expanded_kernel = lp.expand_subst(kernel)
    temp_to_live_interval_start: Dict[str, int] = {}
    temp_to_live_interval_end: Dict[str, int] = {}

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

    icall_kernel_to_just_live_temp_vars: List[Set[str]] = [
        set() for _ in call_kernel_insn_ids]
    icall_kernel_to_just_dead_temp_vars: List[Set[str]] = [
        set() for _ in call_kernel_insn_ids]

    for tv_name, just_alive_idx in temp_to_live_interval_start.items():
        icall_kernel_to_just_live_temp_vars[just_alive_idx].add(tv_name)

    for tv_name, just_dead_idx in temp_to_live_interval_end.items():
        if just_dead_idx != (len(call_kernel_insn_ids) - 1):
            # we ignore the temporaries that died at the last kernel since we cannot
            # reclaim their memory
            icall_kernel_to_just_dead_temp_vars[just_dead_idx+1].add(tv_name)

    # }}}

    new_tvs: Dict[str, lp.TemporaryVariable] = {}
    # a mapping from shape to the available base storages from temp variables
    # that were dead.
    shape_to_available_base_storage: Dict[int, Set[str]] = defaultdict(set)

    for icall_kernel, _ in enumerate(call_kernel_insn_ids):
        just_dead_temps = icall_kernel_to_just_dead_temp_vars[icall_kernel]
        to_be_allocated_temps = icall_kernel_to_just_live_temp_vars[icall_kernel]

        # reclaim base storage from the dead temporaries
        for tv_name in sorted(just_dead_temps):
            tv = new_tvs[tv_name]
            assert tv.base_storage is not None
            assert tv.base_storage not in shape_to_available_base_storage[tv.nbytes]
            shape_to_available_base_storage[tv.nbytes].add(tv.base_storage)

        # assign base storages to 'to_be_allocated_temps'
        for tv_name in sorted(to_be_allocated_temps):
            tv = kernel.temporary_variables[tv_name]
            assert tv.name not in new_tvs
            assert tv.base_storage is None
            if shape_to_available_base_storage[tv.nbytes]:
                base_storage = sorted(shape_to_available_base_storage[tv.nbytes])[0]
                shape_to_available_base_storage[tv.nbytes].remove(base_storage)
            else:
                base_storage = vng("_actx_tmp_base")

            new_tvs[tv.name] = tv.copy(base_storage=base_storage)

    for name, tv in kernel.temporary_variables.items():
        if tv.address_space != AddressSpace.GLOBAL:
            new_tvs[name] = tv
        else:
            pass

    kernel = kernel.copy(temporary_variables=new_tvs)

    old_tmp_mem_requirement = sum(
        tv.nbytes
        for tv in kernel.temporary_variables.values())
    new_tmp_mem_requirement = sum(
        {tv.base_storage: tv.nbytes
         for tv in kernel.temporary_variables.values()}.values())
    split_actx_utils_logger.info(
        "[alias_global_temporaries]: Reduced memory requirement from "
        f"{old_tmp_mem_requirement*1e-6:.1f}MB to"
        f" {new_tmp_mem_requirement*1e-6:.1f}MB.")

    return t_unit.with_kernel(kernel)

# }}}

# vim: fdm=marker
