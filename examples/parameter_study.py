from dataclasses import dataclass

import numpy as np  # for the data types
from pytools.tag import Tag

from arraycontext.parameter_study import (pack_for_parameter_study,
                                          unpack_parameter_study)
from arraycontext.parameter_study.transform import (ParamStudyPytatoPyOpenCLArrayContext,
                                                    ParameterStudyAxisTag)

import pyopencl as cl
import pytato as pt

ctx = cl.create_some_context(interactive=False)
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
    return param1 + param2

@dataclass(frozen=True)
class ParameterStudyForX(ParameterStudyAxisTag):
    pass

@dataclass(frozen=True)
class ParameterStudyForY(ParameterStudyAxisTag):
    pass

# Pack a parameter study of 3 instances for both x and y.
packx = pack_for_parameter_study(actx, ParameterStudyForX, (3,), x, x1, x2)
packy = pack_for_parameter_study(actx, ParameterStudyForY, (4,), y, y1, y2, y3)

compiled_rhs = actx.compile(rhs)  # Build the function caller

# Builds a trace for a single instance of evaluating the RHS and
# then converts it to a program which takes our multiple instances of `x` and `y`.
output = compiled_rhs(packx, packy)
breakpoint()
output_2 = compiled_rhs(x,y)

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
