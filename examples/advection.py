from dataclasses import dataclass

import numpy as np  # for the data types

import pyopencl as cl

from arraycontext.parameter_study import (
    ParameterStudyAxisTag,
    ParamStudyPytatoPyOpenCLArrayContext,
    pack_for_parameter_study,
    unpack_parameter_study,
)


ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
actx = ParamStudyPytatoPyOpenCLArrayContext(queue)


@dataclass(frozen=True)
class ParamStudy1(ParameterStudyAxisTag):
    """
    1st parameter study.
    """


def test_one_time_step_advection():

    seed = 12345
    rng = np.random.default_rng(seed)

    base_shape = np.prod((15, 5))
    x0 = actx.from_numpy(rng.random(base_shape))
    x1 = actx.from_numpy(rng.random(base_shape))
    x2 = actx.from_numpy(rng.random(base_shape))
    x3 = actx.from_numpy(rng.random(base_shape))

    speed_shape = (1,)
    y0 = actx.from_numpy(rng.random(speed_shape))
    y1 = actx.from_numpy(rng.random(speed_shape))
    y2 = actx.from_numpy(rng.random(speed_shape))
    y3 = actx.from_numpy(rng.random(speed_shape))

    ht = 0.0001
    hx = 0.005
    inds = np.arange(base_shape, dtype=int)
    kp1 = actx.from_numpy(np.roll(inds, -1))
    km1 = actx.from_numpy(np.roll(inds, 1))

    def rhs(fields, wave_speed):
        # 2nd order in space finite difference
        return fields + wave_speed * (-1) * (ht / (2 * hx)) * \
                (fields[kp1] - fields[km1])

    pack_x = pack_for_parameter_study(actx, ParamStudy1, (4,), x0, x1, x2, x3)
    breakpoint()
    assert pack_x.shape == (75, 4)

    pack_y = pack_for_parameter_study(actx, ParamStudy1, (4,), y0, y1, y2, y3)
    breakpoint()
    assert pack_y.shape == (1, 4)

    compiled_rhs = actx.compile(rhs)
    breakpoint()

    output = compiled_rhs(pack_x, pack_y)
    breakpoint()
    assert output.shape(75, 4)

    output_x = unpack_parameter_study(output, ParamStudy1)
    assert len(output_x) == 1  # Only 1 study associated with this variable.
    assert len(output_x[0]) == 4  # 4 inputs for the parameter study.

    print("All checks passed")

# Call it.


test_one_time_step_advection()
