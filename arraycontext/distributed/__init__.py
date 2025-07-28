"""
.. autoclass:: DistributedExecutionContext
"""

__copyright__ = """
Copyright (C) 2020-1 University of Illinois Board of Trustees
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

from typing import Any, Hashable, Tuple, TYPE_CHECKING
import numpy as np

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mpi4py import MPI


CommTag = Hashable


# {{{ abstract

class Request(ABC):
    @abstractmethod
    def complete_before(self, req: "Request"):
        pass

    @abstractmethod
    def complete(self) -> None:
        pass


class RecvRequest(Request):
    @abstractmethod
    def complete(self) -> Any:
        pass


class DistributedExecutionContext:
    @property
    @abstractmethod
    def rank(self):
        pass

    @property
    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def compile(self, f):
        pass

    @abstractmethod
    def initiate_send(self, remote_rank: int, comm_tag: CommTag,
            array: Any) -> Request:
        pass

    @abstractmethod
    def initiate_recv(self, remote_rank: int, comm_tag: CommTag,
            shape: Tuple[int, ...], dtype: np.dtype) -> RecvRequest:
        pass

# }}}


# {{{

class _MPIRequest(ABC):
    @abstractmethod
    def complete_before(self, req: "Request"):
        pass

    @abstractmethod
    def complete(self) -> None:
        pass


class _MPIRecvRequest(Request):
    @abstractmethod
    def complete(self) -> Any:
        pass


class MPIDistributedExecutionContext:
    def __init__(self, mpi_communicator: "MPI.Comm") -> None:
        self.mpi_communicator = mpi_communicator

    @property
    @abstractmethod
    def rank(self):
        return self.mpi_communicator.rank

    @property
    @abstractmethod
    def size(self):
        return self.mpi_communicator.size

    def compile(self, f):
        1/0

    @abstractmethod
    def initiate_send(self, remote_rank: int, comm_tag: CommTag,
            array: Any) -> Request:
        pass

    @abstractmethod
    def initiate_recv(self, remote_rank: int, comm_tag: CommTag,
            shape: Tuple[int, ...], dtype: np.dtype) -> RecvRequest:
        pass

# }}}

# vim: foldmethod=marker
