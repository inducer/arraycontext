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


    ht = 0.0001
    hx = 0.005
    inds = np.arange(base_shape, dtype=int)
    kp1 = actx.from_numpy(np.roll(inds, -1))
    km1 = actx.from_numpy(np.roll(inds, 1))

    def rhs(fields, wave_speed):
        # 2nd order in space finite difference
        return fields + wave_speed * (-1) * (ht / (2 * hx)) * \
                (fields[kp1] - fields[km1])

    wave_speeds = [actx.from_numpy(np.random.random(1)) for _ in range(255)]
    print(type(wave_speeds[0])) 
    packed_speeds = pack_for_parameter_study(actx, ParamStudy1, *wave_speeds)

    compiled_rhs = actx.compile(rhs)

    output = compiled_rhs(x0, packed_speeds)
    output = actx.freeze(output)
    
    expanded_output = actx.to_numpy(output).T

    # Now for all the single values.
    for idx in range(len(wave_speeds)):
        out = compiled_rhs(x0, wave_speeds[idx])
        out = actx.freeze(out)
        assert np.allclose(expanded_output[idx], actx.to_numpy(out))


    print("All checks passed")

# Call it.


test_one_time_step_advection()
