import arraycontext
from dataclasses import dataclass

import numpy as np # for the data types
from pytools.tag import Tag

from arraycontext.impl.pytato.__init__ import (PytatoPyOpenCLArrayContextUQ,
                                               PytatoPyOpenCLArrayContext)

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


x = np.random.random((15,5))
x1 = np.random.random((15,5))
x2 = np.random.random((15,5))

y = np.random.random((15,5))
y1 = np.random.random((15,5))
y2 = np.random.random((15,5))


actx = PytatoPyOpenCLArrayContextUQ

out = actx.pack_for_uq(actx,"x", x, x1, x2, "y", y, y1, y2)
print("===============OUT======================")
print(out)

x = out["x"]
y = out["y"]

breakpoint()

x + y

