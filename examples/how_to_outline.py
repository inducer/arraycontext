from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

import pytato as pt
from pytools.obj_array import ObjectArray1D, make_obj_array

from arraycontext import (
    Array,
    PytatoJAXArrayContext as BasePytatoJAXArrayContext,
    dataclass_array_container,
    with_container_arithmetic,
)


if TYPE_CHECKING:
    from arraycontext import (
        ArrayOrArithContainer,
    )


Ncalls = 300


class PytatoJAXArrayContext(BasePytatoJAXArrayContext):
    @override
    def transform_dag(self,
                dag: pt.AbstractResultWithNamedArrays
            ):
        # Test 1: Test that the number of untransformed call sites are as
        # expected
        assert pt.analysis.get_num_call_sites(dag) == Ncalls

        dag = pt.tag_all_calls_to_be_inlined(dag)
        # FIXME: Re-enable this when concatenation is added to pytato
        # print("[Pre-concatenation] Number of nodes =",
        #       pt.analysis.get_num_nodes(pt.inline_calls(dag)))
        # dag = pt.concatenate_calls(
        #     dag,
        #     lambda cs: pt.tags.FunctionIdentifier("foo") in cs.call.function.tags
        # )
        #
        # # Test 2: Test that only one call-sites is left post concatenation
        # assert pt.analysis.get_num_call_sites(dag) == 1
        #
        # dag = pt.inline_calls(dag)
        # print("[Post-concatenation] Number of nodes =",
        #       pt.analysis.get_num_nodes(dag))
        dag = pt.inline_calls(dag)

        return dag


actx = PytatoJAXArrayContext()


@with_container_arithmetic(
    bcast_obj_array=True,
    eq_comparison=False,
    rel_comparison=False,
)
@dataclass_array_container
@dc.dataclass(frozen=True)
class State:
    mass: Array | np.ndarray
    vel: ObjectArray1D[Array]


@actx.outline
def foo(
            x1: ArrayOrArithContainer,
            x2: ArrayOrArithContainer
        ) -> ArrayOrArithContainer:
    return (2*x1 + 3*x2 + x1**3 + x2**4
            + actx.np.minimum(2*x1, 4*x2)
            + actx.np.maximum(7*x1, 8*x2)
        )


rng = np.random.default_rng(0)
Ndof = 10
Ndim = 3

results = []

for _ in range(Ncalls):
    Nel = rng.integers(low=4, high=17)
    state1_np = State(
        mass=rng.random((Nel, Ndof)),
        vel=make_obj_array([*rng.random((Ndim, Nel, Ndof))]),
    )
    state2_np = State(
        mass=rng.random((Nel, Ndof)),
        vel=make_obj_array([*rng.random((Ndim, Nel, Ndof))]),
    )

    state1 = actx.from_numpy(state1_np)
    state2 = actx.from_numpy(state2_np)
    results.append(foo(state1, state2))

actx.to_numpy(make_obj_array(results))
