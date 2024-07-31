__copyright__ = "Copyright (C) 2020-21 University of Illinois Board of Trustees"

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
from typing import Union

import numpy as np
import pytest

from pytools.obj_array import make_obj_array
from pytools.tag import Tag

from arraycontext import (
    ArrayContainer,
    ArrayContext,
    EagerJAXArrayContext,
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext,
    dataclass_array_container,
    deserialize_container,
    pytest_generate_tests_for_array_contexts,
    serialize_container,
    tag_axes,
    with_array_context,
    with_container_arithmetic,
)
from arraycontext.pytest import (
    _PytestEagerJaxArrayContextFactory,
    _PytestNumpyArrayContextFactory,
    _PytestPyOpenCLArrayContextFactoryWithClass,
    _PytestPytatoJaxArrayContextFactory,
    _PytestPytatoPyOpenCLArrayContextFactory,
)


logger = logging.getLogger(__name__)


# {{{ array context fixture

class _PyOpenCLArrayContextForTests(PyOpenCLArrayContext):
    """Like :class:`PyOpenCLArrayContext`, but applies no program transformations
    whatsoever. Only to be used for testing internal to :mod:`arraycontext`.
    """

    def transform_loopy_program(self, t_unit):
        return t_unit


class _PytatoPyOpenCLArrayContextForTests(PytatoPyOpenCLArrayContext):
    """Like :class:`PytatoPyOpenCLArrayContext`, but applies no program
    transformations whatsoever. Only to be used for testing internal to
    :mod:`arraycontext`.
    """

    def transform_loopy_program(self, t_unit):
        return t_unit


class _PyOpenCLArrayContextWithHostScalarsForTestsFactory(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class = _PyOpenCLArrayContextForTests


class _PyOpenCLArrayContextForTestsFactory(
        _PyOpenCLArrayContextWithHostScalarsForTestsFactory):
    force_device_scalars = True


class _PytatoPyOpenCLArrayContextForTestsFactory(
        _PytestPytatoPyOpenCLArrayContextFactory):
    actx_class = _PytatoPyOpenCLArrayContextForTests


pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    _PyOpenCLArrayContextForTestsFactory,
    _PyOpenCLArrayContextWithHostScalarsForTestsFactory,
    _PytatoPyOpenCLArrayContextForTestsFactory,
    _PytestEagerJaxArrayContextFactory,
    _PytestPytatoJaxArrayContextFactory,
    _PytestNumpyArrayContextFactory,
    ])


def _acf():
    import pyopencl as cl

    context = cl._csc()
    queue = cl.CommandQueue(context)
    return _PyOpenCLArrayContextForTests(queue, force_device_scalars=True)

# }}}


# {{{ stand-in DOFArray implementation

@with_container_arithmetic(
        bcast_obj_array=True,
        bcast_numpy_array=True,
        bitwise=True,
        rel_comparison=True,
        _cls_has_array_context_attr=True)
class DOFArray:
    def __init__(self, actx, data):
        if not (actx is None or isinstance(actx, ArrayContext)):
            raise TypeError("actx must be of type ArrayContext")

        if not isinstance(data, tuple):
            raise TypeError("'data' argument must be a tuple")

        self.array_context = actx
        self.data = data

    __array_priority__ = 10

    def __bool__(self):
        if len(self) == 1 and self.data[0].size == 1:
            return bool(self.data[0])

        raise ValueError(
                "The truth value of an array with more than one element is "
                "ambiguous. Use actx.np.any(x) or actx.np.all(x)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __repr__(self):
        return f"DOFArray({self.data!r})"

    @classmethod
    def _serialize_init_arrays_code(cls, instance_name):
        return {"_":
                (f"{instance_name}_i", f"{instance_name}")}

    @classmethod
    def _deserialize_init_arrays_code(cls, template_instance_name, args):
        (_, arg), = args.items()
        # Why tuple([...])? https://stackoverflow.com/a/48592299
        return (f"{template_instance_name}.array_context, tuple([{arg}])")

    @property
    def size(self):
        return sum(ary.size for ary in self.data)

    @property
    def real(self):
        return DOFArray(self.array_context, tuple([subary.real for subary in self]))

    @property
    def imag(self):
        return DOFArray(self.array_context, tuple([subary.imag for subary in self]))


@serialize_container.register(DOFArray)
def _serialize_dof_container(ary: DOFArray):
    return list(enumerate(ary.data))


@deserialize_container.register(DOFArray)
# https://github.com/python/mypy/issues/13040
def _deserialize_dof_container(  # type: ignore[misc]
        template, iterable):
    def _raise_index_inconsistency(i, stream_i):
        raise ValueError(
                "out-of-sequence indices supplied in DOFArray deserialization "
                f"(expected {i}, received {stream_i})")

    return type(template)(
            template.array_context,
            data=tuple(
                v if i == stream_i else _raise_index_inconsistency(i, stream_i)
                for i, (stream_i, v) in enumerate(iterable)))


@with_array_context.register(DOFArray)
# https://github.com/python/mypy/issues/13040
def _with_actx_dofarray(ary: DOFArray, actx: ArrayContext) -> DOFArray:  # type: ignore[misc]
    return type(ary)(actx, ary.data)

# }}}


# {{{ nested containers

@with_container_arithmetic(bcast_obj_array=False,
        eq_comparison=False, rel_comparison=False,
        _cls_has_array_context_attr=True)
@dataclass_array_container
@dataclass(frozen=True)
class MyContainer:
    name: str
    mass: Union[DOFArray, np.ndarray]
    momentum: np.ndarray
    enthalpy: Union[DOFArray, np.ndarray]

    @property
    def array_context(self):
        if isinstance(self.mass, np.ndarray):
            return next(iter(self.mass)).array_context
        else:
            return self.mass.array_context


@with_container_arithmetic(
        bcast_obj_array=False,
        bcast_container_types=(DOFArray, np.ndarray),
        matmul=True,
        rel_comparison=True,
        _cls_has_array_context_attr=True)
@dataclass_array_container
@dataclass(frozen=True)
class MyContainerDOFBcast:
    name: str
    mass: Union[DOFArray, np.ndarray]
    momentum: np.ndarray
    enthalpy: Union[DOFArray, np.ndarray]

    @property
    def array_context(self):
        if isinstance(self.mass, np.ndarray):
            return next(iter(self.mass)).array_context
        else:
            return self.mass.array_context


def _get_test_containers(actx, ambient_dim=2, shapes=50_000):
    from numbers import Number
    if isinstance(shapes, (Number, tuple)):
        shapes = [shapes]

    x = DOFArray(actx, tuple([
        actx.from_numpy(randn(shape, np.float64))
        for shape in shapes]))

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    dataclass_of_dofs = MyContainer(
            name="container",
            mass=x,
            momentum=make_obj_array([x] * ambient_dim),
            enthalpy=x)

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    bcast_dataclass_of_dofs = MyContainerDOFBcast(
            name="container",
            mass=x,
            momentum=make_obj_array([x] * ambient_dim),
            enthalpy=x)

    ary_dof = x
    ary_of_dofs = make_obj_array([x] * ambient_dim)
    mat_of_dofs = np.empty((ambient_dim, ambient_dim), dtype=object)
    for i in np.ndindex(mat_of_dofs.shape):
        mat_of_dofs[i] = x

    return (ary_dof, ary_of_dofs, mat_of_dofs, dataclass_of_dofs,
            bcast_dataclass_of_dofs)

# }}}


# {{{ assert_close_to_numpy*

def randn(shape, dtype):
    rng = np.random.default_rng()
    dtype = np.dtype(dtype)

    if shape == 0:
        ashape = 1
    else:
        ashape = shape

    if dtype.kind == "c":
        dtype = np.dtype(f"<f{dtype.itemsize // 2}")
        r = rng.standard_normal(ashape, dtype) \
            + 1j * rng.standard_normal(ashape, dtype)
    elif dtype.kind == "f":
        r = rng.standard_normal(ashape, dtype)
    elif dtype.kind == "i":
        r = rng.integers(0, 512, ashape, dtype)
    else:
        raise TypeError(dtype.kind)

    if shape == 0:
        return np.array(r[0])

    return r


def assert_close_to_numpy(actx, op, args):
    assert np.allclose(
            actx.to_numpy(
                op(actx.np, *[
                    actx.from_numpy(arg) if isinstance(arg, np.ndarray) else arg
                    for arg in args])),
            op(np, *args))


def assert_close_to_numpy_in_containers(actx, op, args):
    assert_close_to_numpy(actx, op, args)

    ref_result = op(np, *args)

    # {{{ test DOFArrays

    dofarray_args = [
            DOFArray(actx, (actx.from_numpy(arg),))
            if isinstance(arg, np.ndarray) else arg
            for arg in args]

    actx_result = op(actx.np, *dofarray_args)
    if isinstance(actx_result, DOFArray):
        actx_result = actx_result[0]

    assert np.allclose(actx.to_numpy(actx_result), ref_result)

    # }}}

    # {{{ test object arrays of DOFArrays

    obj_array_args = [
            make_obj_array([arg]) if isinstance(arg, DOFArray) else arg
            for arg in dofarray_args]

    obj_array_result = op(actx.np, *obj_array_args)
    if isinstance(obj_array_result, np.ndarray):
        obj_array_result = obj_array_result[0][0]

    assert np.allclose(actx.to_numpy(obj_array_result), ref_result)

    # }}}

# }}}


# {{{ np.function same as numpy

@pytest.mark.parametrize(("sym_name", "n_args", "dtype"), [
            # float only
            ("arctan2", 2, np.float64),
            ("minimum", 2, np.float64),
            ("maximum", 2, np.float64),
            ("where", 3, np.float64),
            ("min", 1, np.float64),
            ("max", 1, np.float64),
            ("any", 1, np.float64),
            ("all", 1, np.float64),
            ("arctan", 1, np.float64),

            # float + complex
            ("sin", 1, np.float64),
            ("sin", 1, np.complex128),
            ("exp", 1, np.float64),
            ("exp", 1, np.complex128),
            ("conj", 1, np.float64),
            ("conj", 1, np.complex128),
            ("vdot", 2, np.float64),
            ("vdot", 2, np.complex128),
            ("abs", 1, np.float64),
            ("abs", 1, np.complex128),
            ("sum", 1, np.float64),
            ("sum", 1, np.complex64),
            ("isnan", 1, np.float64),
            ])
def test_array_context_np_workalike(actx_factory, sym_name, n_args, dtype):
    actx = actx_factory()
    if not hasattr(actx.np, sym_name):
        pytest.skip(f"'{sym_name}' not implemented on '{type(actx).__name__}'")

    ndofs = 512
    args = [randn(ndofs, dtype) for i in range(n_args)]

    c_to_numpy_arc_functions = {
            "atan": "arctan",
            "atan2": "arctan2",
            }

    def evaluate(_np, *_args):
        func = getattr(_np, sym_name,
                getattr(_np, c_to_numpy_arc_functions.get(sym_name, sym_name)))

        return func(*_args)

    assert_close_to_numpy_in_containers(actx, evaluate, args)

    if sym_name in ["where", "min", "max", "any", "all", "conj", "vdot", "sum"]:
        pytest.skip(f"'{sym_name}' not supported on scalars")

    args = [randn(0, dtype)[()] for i in range(n_args)]
    assert_close_to_numpy(actx, evaluate, args)


@pytest.mark.parametrize(("sym_name", "n_args", "dtype"), [
            ("zeros_like", 1, np.float64),
            ("zeros_like", 1, np.complex128),
            ("ones_like", 1, np.float64),
            ("ones_like", 1, np.complex128),
            ])
def test_array_context_np_like(actx_factory, sym_name, n_args, dtype):
    actx = actx_factory()

    ndofs = 512
    args = [randn(ndofs, dtype) for i in range(n_args)]
    assert_close_to_numpy(
            actx, lambda _np, *_args: getattr(_np, sym_name)(*_args), args)

    for c in (42.0, *_get_test_containers(actx)):
        result = getattr(actx.np, sym_name)(c)
        result = actx.thaw(actx.freeze(result))

        if sym_name == "zeros_like":
            if np.isscalar(result):
                assert result == 0.0
            else:
                assert actx.to_numpy(actx.np.all(actx.np.equal(result, 0.0)))
        elif sym_name == "ones_like":
            if np.isscalar(result):
                assert result == 1.0
            else:
                assert actx.to_numpy(actx.np.all(actx.np.equal(result, 1.0)))
        else:
            raise ValueError(f"unknown method: '{sym_name}'")

# }}}


# {{{ array manipulations

def test_actx_stack(actx_factory):
    rng = np.random.default_rng()

    actx = actx_factory()

    ndofs = 5000
    args = [rng.normal(size=ndofs) for i in range(10)]

    assert_close_to_numpy_in_containers(
            actx, lambda _np, *_args: _np.stack(_args), args)


def test_actx_concatenate(actx_factory):
    rng = np.random.default_rng()
    actx = actx_factory()

    ndofs = 5000
    args = [rng.normal(size=ndofs) for i in range(10)]

    assert_close_to_numpy(
            actx, lambda _np, *_args: _np.concatenate(_args), args)


def test_actx_reshape(actx_factory):
    rng = np.random.default_rng()
    actx = actx_factory()

    for new_shape in [(3, 2), (3, -1), (6,), (-1,)]:
        assert_close_to_numpy(
                actx, lambda _np, *_args: _np.reshape(*_args),
                (rng.normal(size=(2, 3)), new_shape))


def test_actx_ravel(actx_factory):
    from numpy.random import default_rng
    actx = actx_factory()
    rng = default_rng()
    ndim = rng.integers(low=1, high=6)
    shape = tuple(rng.integers(2, 7, ndim))

    assert_close_to_numpy(actx, lambda _np, ary: _np.ravel(ary),
                          (rng.random(shape),))

# }}}


# {{{ arithmetic same as numpy

def test_dof_array_arithmetic_same_as_numpy(actx_factory):
    rng = np.random.default_rng()
    actx = actx_factory()

    ndofs = 50_000

    def get_real(ary):
        return ary.real

    def get_imag(ary):
        return ary.imag

    import operator
    from random import randrange, uniform

    from pytools import generate_nonnegative_integer_tuples_below as gnitb
    for op_func, n_args, use_integers in [
            (operator.add, 2, False),
            (operator.sub, 2, False),
            (operator.mul, 2, False),
            (operator.truediv, 2, False),
            (operator.pow, 2, False),
            # FIXME pyopencl.Array doesn't do mod.
            # (operator.mod, 2, True),
            # (operator.mod, 2, False),
            # (operator.imod, 2, True),
            # (operator.imod, 2, False),
            # FIXME: Two outputs
            # (divmod, 2, False),

            (operator.iadd, 2, False),
            (operator.isub, 2, False),
            (operator.imul, 2, False),
            (operator.itruediv, 2, False),

            (operator.and_, 2, True),
            (operator.xor, 2, True),
            (operator.or_, 2, True),

            (operator.iand, 2, True),
            (operator.ixor, 2, True),
            (operator.ior, 2, True),

            (operator.ge, 2, False),
            (operator.lt, 2, False),
            (operator.gt, 2, False),
            (operator.eq, 2, True),
            (operator.ne, 2, True),

            (operator.pos, 1, False),
            (operator.neg, 1, False),
            (operator.abs, 1, False),

            (get_real, 1, False),
            (get_imag, 1, False),
            ]:
        for is_array_flags in gnitb(2, n_args):
            if sum(is_array_flags) == 0:
                # all scalars, no need to test
                continue

            if is_array_flags[0] == 0 and op_func in [
                    operator.iadd, operator.isub,
                    operator.imul, operator.itruediv,
                    operator.iand, operator.ixor, operator.ior,
                    ]:
                # can't do in place operations with a scalar lhs
                continue

            if op_func == operator.ge:
                op_func_actx = actx.np.greater_equal
            elif op_func == operator.lt:
                op_func_actx = actx.np.less
            elif op_func == operator.gt:
                op_func_actx = actx.np.greater
            elif op_func == operator.eq:
                op_func_actx = actx.np.equal
            elif op_func == operator.ne:
                op_func_actx = actx.np.not_equal
            else:
                op_func_actx = op_func

            args = [
                    (0.5+rng.uniform(size=ndofs)
                        if not use_integers else
                        rng.integers(3, 200, size=ndofs))

                    if is_array_flag else
                    (uniform(0.5, 2)
                        if not use_integers
                        else randrange(3, 200))
                    for is_array_flag in is_array_flags]

            # {{{ get reference numpy result

            # make a copy for the in place operators
            ref_args = [
                    arg.copy() if isinstance(arg, np.ndarray) else arg
                    for arg in args]
            ref_result = op_func(*ref_args)

            # }}}

            # {{{ test DOFArrays

            actx_args = [
                    DOFArray(actx, (actx.from_numpy(arg),))
                    if isinstance(arg, np.ndarray) else arg
                    for arg in args]

            actx_result = actx.to_numpy(op_func_actx(*actx_args)[0])

            assert np.allclose(actx_result, ref_result)

            # }}}

            # {{{ test object arrays of DOFArrays

            # It would be very nice if comparisons on object arrays behaved
            # consistently with everything else. Alas, they do not. Instead:
            #
            # 0.5 < obj_array(DOFArray) -> obj_array([True])
            #
            # because hey, 0.5 < DOFArray returned something truthy.

            if op_func not in [
                    operator.eq, operator.ne,
                    operator.le, operator.lt,
                    operator.ge, operator.gt,

                    operator.iadd, operator.isub,
                    operator.imul, operator.itruediv,
                    operator.iand, operator.ixor, operator.ior,

                    # All Python objects are real-valued, right?
                    get_imag,
                    ]:
                obj_array_args = [
                        make_obj_array([arg]) if isinstance(arg, DOFArray) else arg
                        for arg in actx_args]

                obj_array_result = actx.to_numpy(
                        op_func_actx(*obj_array_args)[0][0])

                assert np.allclose(obj_array_result, ref_result)

            # }}}

# }}}


# {{{ reductions same as numpy

@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_reductions_same_as_numpy(actx_factory, op):
    rng = np.random.default_rng()
    actx = actx_factory()

    ary = rng.normal(size=3000)
    np_red = getattr(np, op)(ary)
    actx_red = getattr(actx.np, op)(actx.from_numpy(ary))
    actx_red = actx.to_numpy(actx_red)

    from numbers import Number

    if isinstance(actx, PyOpenCLArrayContext) and (not actx._force_device_scalars):
        assert isinstance(actx_red, Number)
    else:
        assert actx_red.shape == ()

    assert np.allclose(np_red, actx_red)


@pytest.mark.parametrize("sym_name", ["any", "all"])
def test_any_all_same_as_numpy(actx_factory, sym_name):
    actx = actx_factory()
    if not hasattr(actx.np, sym_name):
        pytest.skip(f"'{sym_name}' not implemented on '{type(actx).__name__}'")

    rng = np.random.default_rng()
    ary_any = rng.integers(0, 2, 512)
    ary_all = np.ones(512)

    assert_close_to_numpy_in_containers(actx,
                lambda _np, *_args: getattr(_np, sym_name)(*_args), [ary_any])
    assert_close_to_numpy_in_containers(actx,
                lambda _np, *_args: getattr(_np, sym_name)(*_args), [ary_all])
    assert_close_to_numpy_in_containers(actx,
                lambda _np, *_args: getattr(_np, sym_name)(*_args), [1 - ary_all])


def test_array_equal(actx_factory):
    actx = actx_factory()

    sym_name = "array_equal"
    if not hasattr(actx.np, sym_name):
        pytest.skip(f"'{sym_name}' not implemented on '{type(actx).__name__}'")

    rng = np.random.default_rng()
    ary = rng.integers(0, 2, 512)
    ary_copy = ary.copy()
    ary_diff_values = np.ones(512)
    ary_diff_shape = np.ones(511)
    ary_diff_type = DOFArray(actx, (np.ones(512),))

    # Equal
    assert_close_to_numpy_in_containers(actx,
        lambda _np, *_args: getattr(_np, sym_name)(*_args), [ary, ary_copy])

    # Different values
    assert_close_to_numpy_in_containers(actx,
        lambda _np, *_args: getattr(_np, sym_name)(*_args), [ary, ary_diff_values])

    # Different shapes
    assert_close_to_numpy_in_containers(actx,
        lambda _np, *_args: getattr(_np, sym_name)(*_args), [ary, ary_diff_shape])

    # Different types
    assert not actx.to_numpy(actx.np.array_equal(ary, ary_diff_type))

    # Empty
    ary_empty = np.empty((5, 0), dtype=object)
    ary_empty_copy = ary_empty.copy()
    assert actx.to_numpy(actx.np.array_equal(ary_empty, ary_empty_copy))


# }}}


# {{{ test array context einsum

@pytest.mark.parametrize("spec", [
    "ij->ij",
    "ij->ji",
    "ii->i",
])
def test_array_context_einsum_array_manipulation(actx_factory, spec):
    actx = actx_factory()
    rng = np.random.default_rng()

    mat = actx.from_numpy(rng.normal(size=(10, 10)))
    res = actx.to_numpy(actx.einsum(spec, mat))
    ans = np.einsum(spec, actx.to_numpy(mat))
    assert np.allclose(res, ans)


@pytest.mark.parametrize("spec", [
    "ij,ij->ij",
    "ij,ji->ij",
    "ij,kj->ik",
])
def test_array_context_einsum_array_matmatprods(actx_factory, spec):
    actx = actx_factory()
    rng = np.random.default_rng()

    mat_a = actx.from_numpy(rng.normal(size=(5, 5)))
    mat_b = actx.from_numpy(rng.normal(size=(5, 5)))
    res = actx.to_numpy(actx.einsum(spec, mat_a, mat_b))
    ans = np.einsum(spec, actx.to_numpy(mat_a), actx.to_numpy(mat_b))
    assert np.allclose(res, ans)


@pytest.mark.parametrize("spec", [
    "im,mj,k->ijk"
])
def test_array_context_einsum_array_tripleprod(actx_factory, spec):
    actx = actx_factory()
    rng = np.random.default_rng()

    mat_a = actx.from_numpy(rng.normal(size=(7, 5)))
    mat_b = actx.from_numpy(rng.normal(size=(5, 7)))
    vec = actx.from_numpy(rng.normal(size=(7)))
    res = actx.to_numpy(actx.einsum(spec, mat_a, mat_b, vec))
    ans = np.einsum(spec,
                    actx.to_numpy(mat_a),
                    actx.to_numpy(mat_b),
                    actx.to_numpy(vec))
    assert np.allclose(res, ans)

# }}}


# {{{ array container classes for test


def test_container_map_on_device_scalar(actx_factory):
    actx = actx_factory()

    expected_sizes = [1, 2, 4, 4, 4]
    arys = _get_test_containers(actx, shapes=0)
    arys += (np.pi,)

    from arraycontext import (
        map_array_container,
        map_reduce_array_container,
        rec_map_array_container,
        rec_map_reduce_array_container,
    )

    for size, ary in zip(expected_sizes, arys[:-1]):
        result = map_array_container(lambda x: x, ary)
        assert actx.to_numpy(actx.np.array_equal(result, ary))
        result = rec_map_array_container(lambda x: x, ary)
        assert actx.to_numpy(actx.np.array_equal(result, ary))

        result = map_reduce_array_container(sum, np.size, ary)
        assert result == size
        result = rec_map_reduce_array_container(sum, np.size, ary)
        assert result == size


def test_container_map(actx_factory):
    actx = actx_factory()
    ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs, _bcast_dc_of_dofs = \
            _get_test_containers(actx)

    # {{{ check

    def _check_allclose(f, arg1, arg2, atol=2.0e-14):
        from arraycontext import NotAnArrayContainerError
        try:
            arg1_iterable = serialize_container(arg1)
            arg2_iterable = serialize_container(arg2)
        except NotAnArrayContainerError:
            assert np.linalg.norm(actx.to_numpy(f(arg1) - arg2)) < atol
        else:
            arg1_subarrays = [
                subarray for _, subarray in arg1_iterable]
            arg2_subarrays = [
                subarray for _, subarray in arg2_iterable]
            for subarray1, subarray2 in zip(arg1_subarrays, arg2_subarrays):
                _check_allclose(f, subarray1, subarray2)

    def func(x):
        return x + 1

    from arraycontext import rec_map_array_container
    result = rec_map_array_container(func, 1)
    assert result == 2

    for ary in [ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs]:
        result = rec_map_array_container(func, ary)
        _check_allclose(func, ary, result)

    from arraycontext import mapped_over_array_containers

    @mapped_over_array_containers
    def mapped_func(x):
        return func(x)

    for ary in [ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs]:
        result = mapped_func(ary)
        _check_allclose(func, ary, result)

    @mapped_over_array_containers(leaf_class=DOFArray)
    def check_leaf(x):
        assert isinstance(x, DOFArray)

    for ary in [ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs]:
        check_leaf(ary)

    # }}}


def test_container_multimap(actx_factory):
    actx = actx_factory()
    ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs, _bcast_dc_of_dofs = \
            _get_test_containers(actx)

    # {{{ check

    def _check_allclose(f, arg1, arg2, atol=2.0e-14):
        from arraycontext import NotAnArrayContainerError
        try:
            arg1_iterable = serialize_container(arg1)
            arg2_iterable = serialize_container(arg2)
        except NotAnArrayContainerError:
            assert np.linalg.norm(actx.to_numpy(f(arg1) - arg2)) < atol
        else:
            arg1_subarrays = [
                subarray for _, subarray in arg1_iterable]
            arg2_subarrays = [
                subarray for _, subarray in arg2_iterable]
            for subarray1, subarray2 in zip(arg1_subarrays, arg2_subarrays):
                _check_allclose(f, subarray1, subarray2)

    def func_all_scalar(x, y):
        return x + y

    def func_first_scalar(x, subary):
        return x + subary

    def func_multiple_scalar(a, subary1, b, subary2):
        return a * subary1 + b * subary2

    from arraycontext import rec_multimap_array_container
    result = rec_multimap_array_container(func_all_scalar, 1, 2)
    assert result == 3

    for ary in [ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs]:
        result = rec_multimap_array_container(func_first_scalar, 1, ary)
        _check_allclose(lambda x: 1 + x, ary, result)

        result = rec_multimap_array_container(func_multiple_scalar, 2, ary, 2, ary)
        _check_allclose(lambda x: 4 * x, ary, result)

    from arraycontext import multimapped_over_array_containers

    @multimapped_over_array_containers
    def mapped_func(a, subary1, b, subary2):
        return func_multiple_scalar(a, subary1, b, subary2)

    for ary in [ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs]:
        result = mapped_func(2, ary, 2, ary)
        _check_allclose(lambda x: 4 * x, ary, result)

    @multimapped_over_array_containers(leaf_class=DOFArray)
    def check_leaf(a, subary1, b, subary2):
        assert isinstance(subary1, DOFArray)
        assert isinstance(subary2, DOFArray)

    for ary in [ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs]:
        check_leaf(2, ary, 2, ary)

    with pytest.raises(AssertionError):
        rec_multimap_array_container(func_multiple_scalar, 2, ary_dof, 2, dc_of_dofs)

    # }}}


def test_container_arithmetic(actx_factory):
    actx = actx_factory()
    ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs, bcast_dc_of_dofs = \
            _get_test_containers(actx)

    # {{{ check

    def _check_allclose(f, arg1, arg2, atol=5.0e-14):
        assert np.linalg.norm(actx.to_numpy(f(arg1) - arg2)) < atol

    from functools import partial

    from arraycontext import rec_multimap_array_container
    for ary in [ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs]:
        rec_multimap_array_container(
                partial(_check_allclose, lambda x: 3 * x),
                ary, 2 * ary + ary)
        rec_multimap_array_container(
                partial(_check_allclose, lambda x: actx.np.sin(x)),
                ary, actx.np.sin(ary))

    with pytest.raises(TypeError):
        ary_of_dofs + dc_of_dofs

    with pytest.raises(TypeError):
        dc_of_dofs + ary_of_dofs

    with pytest.raises(TypeError):
        ary_dof + dc_of_dofs

    with pytest.raises(TypeError):
        dc_of_dofs + ary_dof

    bcast_result = ary_dof + bcast_dc_of_dofs
    bcast_dc_of_dofs + ary_dof

    assert actx.to_numpy(actx.np.linalg.norm(bcast_result.mass
                                             - 2*ary_of_dofs)) < 1e-8

    mock_gradient = MyContainerDOFBcast(
            name="yo",
            mass=ary_of_dofs,
            momentum=mat_of_dofs,
            enthalpy=ary_of_dofs)

    grad_matvec_result = mock_gradient @ ary_of_dofs
    assert isinstance(grad_matvec_result.mass, DOFArray)
    assert grad_matvec_result.momentum.shape == ary_of_dofs.shape

    assert actx.to_numpy(actx.np.linalg.norm(
        grad_matvec_result.mass - sum(ary_of_dofs**2)
        )) < 1e-8

    # }}}


def test_container_freeze_thaw(actx_factory):
    actx = actx_factory()
    ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs, _bcast_dc_of_dofs = \
            _get_test_containers(actx)

    # {{{ check

    from arraycontext import (
        get_container_context_opt,
        get_container_context_recursively_opt,
    )

    assert get_container_context_opt(ary_of_dofs) is None
    assert get_container_context_opt(mat_of_dofs) is None
    assert get_container_context_opt(ary_dof) is actx
    assert get_container_context_opt(dc_of_dofs) is actx

    assert get_container_context_recursively_opt(ary_of_dofs) is actx
    assert get_container_context_recursively_opt(mat_of_dofs) is actx

    for ary in [ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs]:
        frozen_ary = actx.freeze(ary)
        thawed_ary = actx.thaw(frozen_ary)
        frozen_ary = actx.freeze(thawed_ary)

        assert get_container_context_recursively_opt(frozen_ary) is None
        assert get_container_context_recursively_opt(thawed_ary) is actx

    actx2 = actx.clone()

    ary_dof_frozen = actx.freeze(ary_dof)
    with pytest.raises(ValueError) as exc_info:
        ary_dof + ary_dof_frozen

    assert "frozen" in str(exc_info.value)

    ary_dof_2 = actx2.thaw(actx.freeze(ary_dof))

    with pytest.raises(ValueError):
        ary_dof + ary_dof_2

    # }}}


@pytest.mark.parametrize("ord", [2, np.inf])
def test_container_norm(actx_factory, ord):
    actx = actx_factory()

    from pytools.obj_array import make_obj_array
    c = MyContainer(name="hey", mass=1, momentum=make_obj_array([2, 3]), enthalpy=5)
    n1 = actx.np.linalg.norm(make_obj_array([c, c]), ord)
    n2 = np.linalg.norm([1, 2, 3, 5]*2, ord)

    assert abs(n1 - n2) < 1e-12

# }}}


# {{{ test flatten and unflatten

@pytest.mark.parametrize("shapes", [
    0,                          # tests device scalars when flattening
    512,
    [(128, 67)],
    [(127, 67), (18, 0)],       # tests 0-sized arrays
    [(64, 7), (154, 12)]
    ])
def test_flatten_array_container(actx_factory, shapes):
    actx = actx_factory()

    from arraycontext import flatten, unflatten
    arys = _get_test_containers(actx, shapes=shapes)

    for ary in arys:
        flat = flatten(ary, actx)
        assert flat.ndim == 1

        ary_roundtrip = unflatten(ary, flat, actx)

        from arraycontext import rec_multimap_reduce_array_container
        assert rec_multimap_reduce_array_container(
                np.prod,
                lambda x, y: x.shape == y.shape,
                ary, ary_roundtrip)

        assert actx.to_numpy(
                actx.np.linalg.norm(ary - ary_roundtrip)
                ) < 1.0e-15

    # {{{ complex to real

    if isinstance(shapes, (int, tuple)):
        shapes = [shapes]

    ary = DOFArray(actx, tuple([
        actx.from_numpy(randn(shape, np.float64))
        for shape in shapes]))

    template = DOFArray(actx, tuple([
        actx.from_numpy(randn(shape, np.complex128))
        for shape in shapes]))

    flat = flatten(ary, actx)
    ary_roundtrip = unflatten(template, flat, actx, strict=False)

    assert actx.to_numpy(
            actx.np.linalg.norm(ary - ary_roundtrip)
            ) < 1.0e-15

    # }}}


def _checked_flatten(ary, actx, leaf_class=None):
    from arraycontext import flat_size_and_dtype, flatten
    result = flatten(ary, actx, leaf_class=leaf_class)

    if leaf_class is None:
        size, dtype = flat_size_and_dtype(ary)
        assert result.shape == (size,)
        assert result.dtype == dtype

    return result


def test_flatten_array_container_failure(actx_factory):
    actx = actx_factory()

    from arraycontext import unflatten
    ary = _get_test_containers(actx, shapes=512)[0]
    flat_ary = _checked_flatten(ary, actx)

    if not isinstance(actx, NumpyArrayContext):
        with pytest.raises(TypeError):
            # cannot unflatten from a numpy array (except for numpy actx)
            unflatten(ary, actx.to_numpy(flat_ary), actx)

    with pytest.raises(ValueError):
        # cannot unflatten non-flat arrays
        unflatten(ary, flat_ary.reshape(2, -1), actx)

    with pytest.raises(ValueError):
        # cannot unflatten partially
        unflatten(ary, flat_ary[:-1], actx)


def test_flatten_with_leaf_class(actx_factory):
    actx = actx_factory()

    arys = _get_test_containers(actx, shapes=512)

    flat = _checked_flatten(arys[0], actx, leaf_class=DOFArray)
    assert isinstance(flat, actx.array_types)
    assert flat.shape == (arys[0].size,)

    flat = _checked_flatten(arys[1], actx, leaf_class=DOFArray)
    assert isinstance(flat, np.ndarray) and flat.dtype == object
    assert all(isinstance(entry, actx.array_types) for entry in flat)
    assert all(entry.shape == (arys[0].size,) for entry in flat)

    flat = _checked_flatten(arys[3], actx, leaf_class=DOFArray)
    assert isinstance(flat, MyContainer)
    assert isinstance(flat.mass, actx.array_types)
    assert flat.mass.shape == (arys[3].mass.size,)
    assert isinstance(flat.enthalpy, actx.array_types)
    assert flat.enthalpy.shape == (arys[3].enthalpy.size,)
    assert all(isinstance(entry, actx.array_types) for entry in flat.momentum)

# }}}


# {{{ test from_numpy and to_numpy

def test_numpy_conversion(actx_factory):
    actx = actx_factory()
    rng = np.random.default_rng()

    nelements = 42
    ac = MyContainer(
            name="test_numpy_conversion",
            mass=rng.uniform(size=(nelements, nelements)),
            momentum=make_obj_array([rng.uniform(size=nelements) for _ in range(3)]),
            enthalpy=np.array(rng.uniform()),
            )

    ac_actx = actx.from_numpy(ac)
    ac_roundtrip = actx.to_numpy(ac_actx)

    assert np.allclose(ac.mass, ac_roundtrip.mass)
    assert np.allclose(ac.momentum[0], ac_roundtrip.momentum[0])

    if not isinstance(actx, NumpyArrayContext):
        from dataclasses import replace
        ac_with_cl = replace(ac, enthalpy=ac_actx.mass)
        with pytest.raises(TypeError):
            actx.from_numpy(ac_with_cl)

        with pytest.raises(TypeError):
            actx.from_numpy(ac_actx)

        with pytest.raises(TypeError):
            actx.to_numpy(ac)

# }}}


# {{{ test actx.np.linalg.norm

@pytest.mark.parametrize("norm_ord", [2, np.inf])
def test_norm_complex(actx_factory, norm_ord):
    actx = actx_factory()
    a = randn(2000, np.complex128)

    norm_a_ref = np.linalg.norm(a, norm_ord)
    norm_a = actx.np.linalg.norm(actx.from_numpy(a), norm_ord)

    norm_a = actx.to_numpy(norm_a)

    assert abs(norm_a_ref - norm_a)/norm_a < 1e-13


@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
def test_norm_ord_none(actx_factory, ndim):
    actx = actx_factory()

    from numpy.random import default_rng

    rng = default_rng()
    shape = tuple(rng.integers(2, 7, ndim))

    a = rng.random(shape)

    norm_a_ref = np.linalg.norm(a, ord=None)
    norm_a = actx.np.linalg.norm(actx.from_numpy(a), ord=None)

    np.testing.assert_allclose(actx.to_numpy(norm_a), norm_a_ref)

# }}}


# {{{ test_actx_compile helpers

@with_container_arithmetic(bcast_obj_array=True, rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class Velocity2D:
    u: ArrayContainer
    v: ArrayContainer
    array_context: ArrayContext


@with_array_context.register(Velocity2D)
# https://github.com/python/mypy/issues/13040
def _with_actx_velocity_2d(ary, actx):  # type: ignore[misc]
    return type(ary)(ary.u, ary.v, actx)


def scale_and_orthogonalize(alpha, vel):
    from arraycontext import rec_map_array_container
    actx = vel.array_context
    scaled_vel = rec_map_array_container(lambda x: alpha * x,
                                         vel)
    return Velocity2D(-scaled_vel.v, scaled_vel.u, actx)


def test_actx_compile(actx_factory):
    actx = actx_factory()
    rng = np.random.default_rng()

    compiled_rhs = actx.compile(scale_and_orthogonalize)

    v_x = rng.uniform(size=10)
    v_y = rng.uniform(size=10)

    vel = actx.from_numpy(Velocity2D(v_x, v_y, actx))

    scaled_speed = compiled_rhs(np.float64(3.14), vel)

    result = actx.to_numpy(scaled_speed)
    np.testing.assert_allclose(result.u, -3.14*v_y)
    np.testing.assert_allclose(result.v, 3.14*v_x)


def test_actx_compile_python_scalar(actx_factory):
    actx = actx_factory()
    rng = np.random.default_rng()

    compiled_rhs = actx.compile(scale_and_orthogonalize)

    v_x = rng.uniform(size=10)
    v_y = rng.uniform(size=10)

    vel = actx.from_numpy(Velocity2D(v_x, v_y, actx))

    scaled_speed = compiled_rhs(3.14, vel)

    result = actx.to_numpy(scaled_speed)
    np.testing.assert_allclose(result.u, -3.14*v_y)
    np.testing.assert_allclose(result.v, 3.14*v_x)


def test_actx_compile_kwargs(actx_factory):
    actx = actx_factory()
    rng = np.random.default_rng()

    compiled_rhs = actx.compile(scale_and_orthogonalize)

    v_x = rng.uniform(size=10)
    v_y = rng.uniform(size=10)

    vel = actx.from_numpy(Velocity2D(v_x, v_y, actx))

    scaled_speed = compiled_rhs(3.14, vel=vel)

    result = actx.to_numpy(scaled_speed)
    np.testing.assert_allclose(result.u, -3.14*v_y)
    np.testing.assert_allclose(result.v, 3.14*v_x)


def test_actx_compile_with_tuple_output_keys(actx_factory):
    # arraycontext.git<=3c9aee68 would fail due to a bug in output
    # key stringification logic.
    from arraycontext import from_numpy, to_numpy
    actx = actx_factory()
    rng = np.random.default_rng()

    def my_rhs(scale, vel):
        result = np.empty((1, 1), dtype=object)
        result[0, 0] = scale_and_orthogonalize(scale, vel)
        return result

    compiled_rhs = actx.compile(my_rhs)

    v_x = rng.uniform(size=10)
    v_y = rng.uniform(size=10)

    vel = from_numpy(Velocity2D(v_x, v_y, actx), actx)

    scaled_speed = compiled_rhs(3.14, vel=vel)

    result = to_numpy(scaled_speed, actx)[0, 0]
    np.testing.assert_allclose(result.u, -3.14*v_y)
    np.testing.assert_allclose(result.v, 3.14*v_x)

# }}}


# {{{ test_container_equality

def test_container_equality(actx_factory):
    actx = actx_factory()

    ary_dof, _, _, _dc_of_dofs, bcast_dc_of_dofs = \
            _get_test_containers(actx)
    _, _, _, _dc_of_dofs_2, bcast_dc_of_dofs_2 = \
            _get_test_containers(actx)

    # MyContainer sets eq_comparison to False, so equality comparison should
    # not succeed.
    dc = MyContainer(name="yoink", mass=ary_dof, momentum=None, enthalpy=None)
    dc2 = MyContainer(name="yoink", mass=ary_dof, momentum=None, enthalpy=None)
    assert dc != dc2

    assert isinstance(actx.np.equal(bcast_dc_of_dofs, bcast_dc_of_dofs_2),
                      MyContainerDOFBcast)

# }}}


# {{{ test_leaf_array_type_broadcasting

@with_container_arithmetic(
    bcast_obj_array=True,
    bcast_numpy_array=True,
    rel_comparison=True,
    _cls_has_array_context_attr=True)
@dataclass_array_container
@dataclass(frozen=True)
class Foo:
    u: DOFArray

    __array_priority__ = 1  # disallow numpy arithmetic to take precedence

    @property
    def array_context(self):
        return self.u.array_context


def test_leaf_array_type_broadcasting(actx_factory):
    # test support for https://github.com/inducer/arraycontext/issues/49
    actx = actx_factory()

    foo = Foo(DOFArray(actx, (actx.np.zeros(3, dtype=np.float64) + 41, )))
    bar = foo + 4
    baz = foo + actx.from_numpy(4*np.ones((3, )))
    qux = actx.from_numpy(4*np.ones((3, ))) + foo

    np.testing.assert_allclose(actx.to_numpy(bar.u[0]),
                               actx.to_numpy(baz.u[0]))

    np.testing.assert_allclose(actx.to_numpy(bar.u[0]),
                               actx.to_numpy(qux.u[0]))

    def _actx_allows_scalar_broadcast(actx):
        if not isinstance(actx, PyOpenCLArrayContext):
            return True
        else:
            import pyopencl as cl

            # See https://github.com/inducer/pyopencl/issues/498
            return cl.version.VERSION > (2021, 2, 5)

    if _actx_allows_scalar_broadcast(actx):
        quux = foo + actx.from_numpy(np.array(4))
        quuz = actx.from_numpy(np.array(4)) + foo

        np.testing.assert_allclose(actx.to_numpy(bar.u[0]),
                                   actx.to_numpy(quux.u[0]))

        np.testing.assert_allclose(actx.to_numpy(bar.u[0]),
                                   actx.to_numpy(quuz.u[0]))

# }}}


# {{{ test outer product

def test_outer(actx_factory):
    actx = actx_factory()

    a_dof, a_ary_of_dofs, _, _, a_bcast_dc_of_dofs = _get_test_containers(actx)

    b_dof = a_dof + 1
    b_ary_of_dofs = a_ary_of_dofs + 1
    b_bcast_dc_of_dofs = a_bcast_dc_of_dofs + 1

    from arraycontext import outer

    def equal(a, b):
        return actx.to_numpy(actx.np.array_equal(a, b))

    # Two scalars
    assert equal(outer(a_dof, b_dof), a_dof*b_dof)

    # Scalar and vector
    assert equal(outer(a_dof, b_ary_of_dofs), a_dof*b_ary_of_dofs)

    # Vector and scalar
    assert equal(outer(a_ary_of_dofs, b_dof), a_ary_of_dofs*b_dof)

    # Two vectors
    assert equal(
        outer(a_ary_of_dofs, b_ary_of_dofs),
        np.outer(a_ary_of_dofs, b_ary_of_dofs))

    # Scalar and array container
    assert equal(
        outer(a_dof, b_bcast_dc_of_dofs),
        a_dof*b_bcast_dc_of_dofs)

    # Array container and scalar
    assert equal(
        outer(a_bcast_dc_of_dofs, b_dof),
        a_bcast_dc_of_dofs*b_dof)

    # Vector and array container
    assert equal(
        outer(a_ary_of_dofs, b_bcast_dc_of_dofs),
        make_obj_array([a_i*b_bcast_dc_of_dofs for a_i in a_ary_of_dofs]))

    # Array container and vector
    assert equal(
        outer(a_bcast_dc_of_dofs, b_ary_of_dofs),
        MyContainerDOFBcast(
            name="container",
            mass=a_bcast_dc_of_dofs.mass*b_ary_of_dofs,
            momentum=np.outer(a_bcast_dc_of_dofs.momentum, b_ary_of_dofs),
            enthalpy=a_bcast_dc_of_dofs.enthalpy*b_ary_of_dofs))

    # Two array containers
    assert equal(
        outer(a_bcast_dc_of_dofs, b_bcast_dc_of_dofs),
        MyContainerDOFBcast(
            name="container",
            mass=a_bcast_dc_of_dofs.mass*b_bcast_dc_of_dofs.mass,
            momentum=np.outer(
                a_bcast_dc_of_dofs.momentum,
                b_bcast_dc_of_dofs.momentum),
            enthalpy=a_bcast_dc_of_dofs.enthalpy*b_bcast_dc_of_dofs.enthalpy))

    # Non-object numpy arrays should be treated as scalars
    ary_of_floats = np.ones(len(b_bcast_dc_of_dofs.mass))
    assert equal(
        outer(ary_of_floats, b_bcast_dc_of_dofs),
        ary_of_floats*b_bcast_dc_of_dofs)
    assert equal(
        outer(a_bcast_dc_of_dofs, ary_of_floats),
        a_bcast_dc_of_dofs*ary_of_floats)

# }}}


# {{{ test_array_container_with_numpy

@with_container_arithmetic(bcast_obj_array=True, rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class ArrayContainerWithNumpy:
    u: np.ndarray
    v: DOFArray


def test_array_container_with_numpy(actx_factory):
    actx = actx_factory()

    mystate = ArrayContainerWithNumpy(
            u=np.zeros(10),
            v=DOFArray(actx, (actx.from_numpy(np.zeros(42)),)),
            )

    from arraycontext import rec_map_array_container
    rec_map_array_container(lambda x: x, mystate)


# }}}


# {{{ test_actx_compile_on_pure_array_return

def test_actx_compile_on_pure_array_return(actx_factory):
    def _twice(x):
        return 2 * x

    actx = actx_factory()
    ones = actx.thaw(actx.freeze(
        actx.np.zeros(shape=(10, 4), dtype=np.float64) + 1
        ))
    np.testing.assert_allclose(actx.to_numpy(_twice(ones)),
                               actx.to_numpy(actx.compile(_twice)(ones)))

# }}}


# {{{ test_taggable_cl_array_tags

@dataclass(frozen=True)
class MySampleTag(Tag):
    pass


def test_taggable_cl_array_tags(actx_factory):
    actx = actx_factory()
    if not isinstance(actx, PyOpenCLArrayContext):
        pytest.skip(f"not relevant for '{type(actx).__name__}'")

    import pyopencl.array as cl_array
    ary = cl_array.to_device(actx.queue, np.zeros((32, 7)))

    # {{{ check tags are set

    from arraycontext.impl.pyopencl.taggable_cl_array import to_tagged_cl_array
    tagged_ary = to_tagged_cl_array(ary, axes=None,
                                    tags=frozenset((MySampleTag(),)))

    assert tagged_ary.base_data is ary.base_data
    assert tagged_ary.tags == frozenset((MySampleTag(),))

    # }}}

    # {{{ check tags are appended

    from arraycontext import ElementwiseMapKernelTag
    tagged_ary = to_tagged_cl_array(tagged_ary, axes=None,
                                    tags=frozenset((ElementwiseMapKernelTag(),)))

    assert tagged_ary.base_data is ary.base_data
    assert tagged_ary.tags == frozenset(
        (MySampleTag(), ElementwiseMapKernelTag())
    )

    # }}}

    # {{{ test copied tags

    copy_tagged_ary = tagged_ary.copy()

    assert copy_tagged_ary.tags == tagged_ary.tags
    assert copy_tagged_ary.axes == tagged_ary.axes
    assert copy_tagged_ary.base_data != tagged_ary.base_data

    # }}}

# }}}


def test_to_numpy_on_frozen_arrays(actx_factory):
    # See https://github.com/inducer/arraycontext/issues/159
    actx = actx_factory()
    u = actx.freeze(actx.np.zeros(10, dtype="float64")+1)
    np.testing.assert_allclose(actx.to_numpy(u), 1)
    np.testing.assert_allclose(actx.to_numpy(u), 1)


def test_tagging(actx_factory):
    actx = actx_factory()

    if isinstance(actx, (NumpyArrayContext, EagerJAXArrayContext)):
        pytest.skip(f"{type(actx)} has no tagging support")

    from pytools.tag import Tag

    class ExampleTag(Tag):
        pass

    ary = tag_axes(actx, {0: ExampleTag()},
            actx.tag(
                ExampleTag(),
                actx.np.zeros((20, 20), dtype=np.float64)))

    assert ary.tags_of_type(ExampleTag)
    assert ary.axes[0].tags_of_type(ExampleTag)
    assert not ary.axes[1].tags_of_type(ExampleTag)


def test_compile_anonymous_function(actx_factory):
    from functools import partial

    # See https://github.com/inducer/grudge/issues/287
    actx = actx_factory()
    f = actx.compile(lambda x: 2*x+40)
    np.testing.assert_allclose(
        actx.to_numpy(f(1+actx.np.zeros((10, 4), "float64"))),
        42)
    f = actx.compile(partial(lambda x: 2*x+40))
    np.testing.assert_allclose(
        actx.to_numpy(f(1+actx.np.zeros((10, 4), "float64"))),
        42)


@pytest.mark.parametrize(
        ("args", "kwargs"), [
            ((1, 2, 10), {}),
            ((1, 2, 10), {"endpoint": False}),
            ((1, 2, 10), {"endpoint": True}),
            ((2, -3, 20), {}),
            ((1, 5j, 20), {"dtype": np.complex128}),
            ((1, 5, 20), {"dtype": np.complex128}),
            ((1, 5, 20), {"dtype": np.int32}),
            ])
def test_linspace(actx_factory, args, kwargs):
    if "Jax" in actx_factory.__class__.__name__:
        pytest.xfail("jax actx does not have arange")

    actx = actx_factory()

    actx_linspace = actx.to_numpy(actx.np.linspace(*args, **kwargs))
    np_linspace = np.linspace(*args, **kwargs)

    assert np.allclose(actx_linspace, np_linspace)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
