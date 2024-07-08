import arraycontext
from dataclasses import dataclass

import numpy as np # for the data types
from pytools.tag import Tag

from arraycontext.impl.pytato.__init__ import (PytatoPyOpenCLArrayContext)

# The goal of this file is to propagate the uncertainty in an array to the output.


my_context = arraycontext.impl.pytato.PytatoPyOpenCLArrayContext
a = my_context.zeros(my_context, shape=(5,5), dtype=np.int32) + 2

b = my_context.zeros(my_context, (5,5), np.int32) + 15

print(a)
print("========================================================")
print(b)
print("========================================================")

# Eq: z = x + y
# Assumptions: x and y are independently uncertain.


base_shape = (15, 5)
x = np.random.random(base_shape)
x1 = np.random.random(base_shape)
x2 = np.random.random(base_shape)

y = np.random.random(base_shape)
y1 = np.random.random(base_shape)
y2 = np.random.random(base_shape)
y3 = np.random.random(base_shape)


from arraycontext.parameter_study import (pack_for_parameter_study,
                                          ParamStudyPytatoPyOpenCLArrayContext, unpack_parameter_study)
import pyopencl as cl

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

actx = ParamStudyPytatoPyOpenCLArrayContext(queue)

# Pack a parameter study of 3 instances for both x and y.
# We are assuming these are distinct parameter studies.
packx = pack_for_parameter_study(actx,"x",tuple([3]), x, x1, x2)
packy = pack_for_parameter_study(actx,"y",tuple([4]), y, y1, y2, y3)
output_x = unpack_parameter_study(packx, "x")

print(packx)
breakpoint()


def rhs(param1, param2):
    return param1 + param2

compiled_rhs = actx.compile(rhs) # Build the function caller

# Builds a trace for a single instance of evaluating the RHS and then converts it to
# a program which takes our multiple instances of `x` and `y`.
output = compiled_rhs(packx, packy)

assert output.shape == (3,4,15,5) # Distinct parameter studies.

output_x = unpack_parameter_study(output, "x")
output_y = unpack_parameter_study(output, "y")
assert len(output_x) == 3
assert output_x[0].shape == (4,15,5)
assert len(output_y) == 4
assert output_y[0].shape == (3,15,5)
