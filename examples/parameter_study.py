from dataclasses import dataclass

import numpy as np  # for the data types

import pyopencl as cl

from arraycontext.parameter_study import (
    pack_for_parameter_study,
    unpack_parameter_study,
    ParamStudyPytatoPyOpenCLArrayContext,
    ParameterStudyAxisTag,
)


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
actx = ParamStudyPytatoPyOpenCLArrayContext(queue)

# Experimental setup
base_shape = (15, 5)
seed = 12345
rng = np.random.default_rng(seed)
x = actx.from_numpy(rng.random(base_shape))
x1 = actx.from_numpy(rng.random(base_shape))
x2 = actx.from_numpy(rng.random(base_shape))

y = actx.from_numpy(rng.random(base_shape))
y1 = actx.from_numpy(rng.random(base_shape))
y2 = actx.from_numpy(rng.random(base_shape))
y3 = actx.from_numpy(rng.random(base_shape))


# Eq: z = x + y
# Assumptions: x and y are undergoing independent parameter studies.
def rhs(param1, param2):
    import pytato as pt
    return pt.matmul(param1, param2.T)
    return pt.stack([param1[0], param2[10]], axis=0)
    return param1[0] + param2[10]


@dataclass(frozen=True)
class ParameterStudyForX(ParameterStudyAxisTag):
    pass


@dataclass(frozen=True)
class ParameterStudyForY(ParameterStudyAxisTag):
    pass

# Pack a parameter study of 3 instances for x and and 4 instances for y.


packx = pack_for_parameter_study(actx, ParameterStudyForX, (3,), x, x1, x2)
packy = pack_for_parameter_study(actx, ParameterStudyForY, (4,), y, y1, y2, y3)

compiled_rhs = actx.compile(rhs)  # Build the function caller

# Builds a trace for a single instance of evaluating the RHS and
# then converts it to a program which takes our multiple instances of `x` and `y`.
output = compiled_rhs(packx, packy)
output_2 = compiled_rhs(x, y)
breakpoint()

assert output.shape == (15, 5, 3, 4)  # Distinct parameter studies.

output_x = unpack_parameter_study(output, ParameterStudyForX)
output_y = unpack_parameter_study(output, ParameterStudyForY)
assert len(output_x) == 1  # Number of parameter studies involving "x"
assert len(output_x[0]) == 3  # Number of inputs in the 0th parameter study
# All outputs across every other parameter study.
assert output_x[0][0].shape == (15, 5, 4)
assert len(output_y) == 1
assert len(output_y[0]) == 4
assert output_y[0][0].shape == (15, 5, 3)
