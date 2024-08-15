from dataclasses import dataclass

import numpy as np  # for the data types

import pyopencl as cl

from arraycontext.parameter_study import (
    ParameterStudyAxisTag,
    ParamStudyPytatoPyOpenCLArrayContext,
    pack_for_parameter_study,
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


# Eq: z = x @ y.T
# Assumptions: x and y are undergoing independent parameter studies.
# x and y are matrices such that x @ y.T works in the single instance case.
def rhs(param1, param2):
    return param1 @ param2.T


@dataclass(frozen=True)
class ParameterStudyForX(ParameterStudyAxisTag):
    pass


@dataclass(frozen=True)
class ParameterStudyForY(ParameterStudyAxisTag):
    pass

# Pack a parameter study of 3 instances for x and and 4 instances for y.


packx = pack_for_parameter_study(actx, ParameterStudyForX, x, x1, x2)
packy = pack_for_parameter_study(actx, ParameterStudyForY, y, y1, y2, y3)

compiled_rhs = actx.compile(rhs)  # Build the function caller

# Builds a trace for a single instance of evaluating the RHS and
# then converts it to a program which takes our multiple instances of `x` and `y`.
output = compiled_rhs(packx, packy)
output_2 = compiled_rhs(x, y)

numpy_output = actx.to_numpy(output)

assert numpy_output.shape == (15, 15, 3, 4)

out = actx.to_numpy(compiled_rhs(x, y))
assert np.allclose(numpy_output[..., 0, 0], out)

out = actx.to_numpy(compiled_rhs(x, y1))
assert np.allclose(numpy_output[..., 0, 1], out)

out = actx.to_numpy(compiled_rhs(x, y2))
assert np.allclose(numpy_output[..., 0, 2], out)

out = actx.to_numpy(compiled_rhs(x, y3))
assert np.allclose(numpy_output[..., 0, 3], out)

out = actx.to_numpy(compiled_rhs(x1, y))
assert np.allclose(numpy_output[..., 1, 0], out)

out = actx.to_numpy(compiled_rhs(x1, y1))
assert np.allclose(numpy_output[..., 1, 1], out)

out = actx.to_numpy(compiled_rhs(x1, y2))
assert np.allclose(numpy_output[..., 1, 2], out)

out = actx.to_numpy(compiled_rhs(x1, y3))
assert np.allclose(numpy_output[..., 1, 3], out)

out = actx.to_numpy(compiled_rhs(x2, y))
assert np.allclose(numpy_output[..., 2, 0], out)

out = actx.to_numpy(compiled_rhs(x2, y1))
assert np.allclose(numpy_output[..., 2, 1], out)

out = actx.to_numpy(compiled_rhs(x2, y2))
assert np.allclose(numpy_output[..., 2, 2], out)

out = actx.to_numpy(compiled_rhs(x2, y3))
assert np.allclose(numpy_output[..., 2, 3], out)

print("All tests passed!")
