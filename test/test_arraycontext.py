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

from dataclasses import dataclass
import numpy as np
import pytest

from pytools.obj_array import make_obj_array

from arraycontext import (
        ArrayContext,
        dataclass_array_container, with_container_arithmetic,
        serialize_container, deserialize_container,
        freeze, thaw,
        FirstAxisIsElementsTag,
        PyOpenCLArrayContext,
        PytatoPyOpenCLArrayContext,
        ArrayContainer,)
from arraycontext import (  # noqa: F401
        pytest_generate_tests_for_array_contexts,
        )
from arraycontext.pytest import (_PytestPyOpenCLArrayContextFactoryWithClass,
                                 _PytestPytatoPyOpenCLArrayContextFactory)


import logging
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
        return f"DOFArray({repr(self.data)})"

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
    def real(self):
        return DOFArray(self.array_context, tuple([subary.real for subary in self]))

    @property
    def imag(self):
        return DOFArray(self.array_context, tuple([subary.imag for subary in self]))


@serialize_container.register(DOFArray)
def _serialize_dof_container(ary: DOFArray):
    return list(enumerate(ary.data))


@deserialize_container.register(DOFArray)
def _deserialize_dof_container(
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


@freeze.register(DOFArray)
def _freeze_dofarray(ary, actx=None):
    assert actx is None
    return type(ary)(
        None,
        tuple(ary.array_context.freeze(subary) for subary in ary.data))


@thaw.register(DOFArray)
def _thaw_dofarray(ary, actx):
    if ary.array_context is not None:
        raise ValueError("cannot thaw DOFArray that already has an array context")

    return type(ary)(
        actx,
        tuple(actx.thaw(subary) for subary in ary.data))

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
            ])
def test_array_context_np_workalike(actx_factory, sym_name, n_args, dtype):
    actx = actx_factory()
    if not hasattr(actx.np, sym_name):
        pytest.skip(f"'{sym_name}' not implemented on '{type(actx).__name__}'")

    ndofs = 512
    args = [randn(ndofs, dtype) for i in range(n_args)]

    assert_close_to_numpy_in_containers(
            actx, lambda _np, *_args: getattr(_np, sym_name)(*_args), args)


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

# }}}


# {{{ array manipulations

def test_actx_stack(actx_factory):
    actx = actx_factory()

    ndofs = 5000
    args = [np.random.randn(ndofs) for i in range(10)]

    assert_close_to_numpy_in_containers(
            actx, lambda _np, *_args: _np.stack(_args), args)


def test_actx_concatenate(actx_factory):
    actx = actx_factory()

    ndofs = 5000
    args = [np.random.randn(ndofs) for i in range(10)]

    assert_close_to_numpy(
            actx, lambda _np, *_args: _np.concatenate(_args), args)


def test_actx_reshape(actx_factory):
    actx = actx_factory()

    for new_shape in [(3, 2), (3, -1), (6,), (-1,)]:
        assert_close_to_numpy(
                actx, lambda _np, *_args: _np.reshape(*_args),
                (np.random.randn(2, 3), new_shape))


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
    actx = actx_factory()

    ndofs = 50_000

    def get_real(ary):
        return ary.real

    def get_imag(ary):
        return ary.imag

    import operator
    from pytools import generate_nonnegative_integer_tuples_below as gnitb
    from random import uniform, randrange
    for op_func, n_args, use_integers in [
            (operator.add, 2, False),
            (operator.sub, 2, False),
            (operator.mul, 2, False),
            (operator.truediv, 2, False),
            (operator.pow, 2, False),
            # FIXME pyopencl.Array doesn't do mod.
            #(operator.mod, 2, True),
            #(operator.mod, 2, False),
            #(operator.imod, 2, True),
            #(operator.imod, 2, False),
            # FIXME: Two outputs
            #(divmod, 2, False),

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
                    (0.5+np.random.rand(ndofs)
                        if not use_integers else
                        np.random.randint(3, 200, ndofs))

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
    actx = actx_factory()

    ary = np.random.randn(3000)
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


def test_array_equal_same_as_numpy(actx_factory):
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


# }}}


# {{{ test array context einsum

@pytest.mark.parametrize("spec", [
    "ij->ij",
    "ij->ji",
    "ii->i",
])
def test_array_context_einsum_array_manipulation(actx_factory, spec):
    actx = actx_factory()

    mat = actx.from_numpy(np.random.randn(10, 10))
    res = actx.to_numpy(actx.einsum(spec, mat,
                                    tagged=(FirstAxisIsElementsTag())))
    ans = np.einsum(spec, actx.to_numpy(mat))
    assert np.allclose(res, ans)


@pytest.mark.parametrize("spec", [
    "ij,ij->ij",
    "ij,ji->ij",
    "ij,kj->ik",
])
def test_array_context_einsum_array_matmatprods(actx_factory, spec):
    actx = actx_factory()

    mat_a = actx.from_numpy(np.random.randn(5, 5))
    mat_b = actx.from_numpy(np.random.randn(5, 5))
    res = actx.to_numpy(actx.einsum(spec, mat_a, mat_b,
                                    tagged=(FirstAxisIsElementsTag())))
    ans = np.einsum(spec, actx.to_numpy(mat_a), actx.to_numpy(mat_b))
    assert np.allclose(res, ans)


@pytest.mark.parametrize("spec", [
    "im,mj,k->ijk"
])
def test_array_context_einsum_array_tripleprod(actx_factory, spec):
    actx = actx_factory()

    mat_a = actx.from_numpy(np.random.randn(7, 5))
    mat_b = actx.from_numpy(np.random.randn(5, 7))
    vec = actx.from_numpy(np.random.randn(7))
    res = actx.to_numpy(actx.einsum(spec, mat_a, mat_b, vec,
                                    tagged=(FirstAxisIsElementsTag())))
    ans = np.einsum(spec,
                    actx.to_numpy(mat_a),
                    actx.to_numpy(mat_b),
                    actx.to_numpy(vec))
    assert np.allclose(res, ans)

# }}}


# {{{ array container classes for test

@with_container_arithmetic(bcast_obj_array=False,
        eq_comparison=False, rel_comparison=False)
@dataclass_array_container
@dataclass(frozen=True)
class MyContainer:
    name: str
    mass: DOFArray
    momentum: np.ndarray
    enthalpy: DOFArray

    @property
    def array_context(self):
        return self.mass.array_context


@with_container_arithmetic(
        bcast_obj_array=False,
        bcast_container_types=(DOFArray, np.ndarray),
        matmul=True,
        rel_comparison=True,)
@dataclass_array_container
@dataclass(frozen=True)
class MyContainerDOFBcast:
    name: str
    mass: DOFArray
    momentum: np.ndarray
    enthalpy: DOFArray

    @property
    def array_context(self):
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


def test_container_scalar_map(actx_factory):
    actx = actx_factory()

    arys = _get_test_containers(actx, shapes=0)
    arys += (np.pi,)

    from arraycontext import (
            map_array_container, rec_map_array_container,
            map_reduce_array_container, rec_map_reduce_array_container,
            )

    for ary in arys:
        result = map_array_container(lambda x: x, ary)
        assert result is not None
        result = rec_map_array_container(lambda x: x, ary)
        assert result is not None

        result = map_reduce_array_container(np.shape, lambda x: x, ary)
        assert result is not None
        result = rec_map_reduce_array_container(np.shape, lambda x: x, ary)
        assert result is not None


def test_container_multimap(actx_factory):
    actx = actx_factory()
    ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs, bcast_dc_of_dofs = \
            _get_test_containers(actx)

    # {{{ check

    def _check_allclose(f, arg1, arg2, atol=2.0e-14):
        assert np.linalg.norm(actx.to_numpy(f(arg1) - arg2)) < atol

    def func_all_scalar(x, y):
        return x + y

    def func_first_scalar(x, subary):
        return x + subary

    def func_multiple_scalar(a, subary1, b, subary2):
        return a * subary1 + b * subary2

    from arraycontext import rec_multimap_array_container
    result = rec_multimap_array_container(func_all_scalar, 1, 2)
    assert result == 3

    from functools import partial
    for ary in [ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs]:
        result = rec_multimap_array_container(func_first_scalar, 1, ary)
        rec_multimap_array_container(
                partial(_check_allclose, lambda x: 1 + x),
                ary, result)

        result = rec_multimap_array_container(func_multiple_scalar, 2, ary, 2, ary)
        rec_multimap_array_container(
                partial(_check_allclose, lambda x: 4 * x),
                ary, result)

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
    ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs, bcast_dc_of_dofs = \
            _get_test_containers(actx)

    # {{{ check

    from arraycontext import get_container_context
    from arraycontext import get_container_context_recursively

    assert get_container_context(ary_of_dofs) is None
    assert get_container_context(mat_of_dofs) is None
    assert get_container_context(ary_dof) is actx
    assert get_container_context(dc_of_dofs) is actx

    assert get_container_context_recursively(ary_of_dofs) is actx
    assert get_container_context_recursively(mat_of_dofs) is actx

    for ary in [ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs]:
        frozen_ary = freeze(ary)
        thawed_ary = thaw(frozen_ary, actx)
        frozen_ary = freeze(thawed_ary)

        assert get_container_context_recursively(frozen_ary) is None
        assert get_container_context_recursively(thawed_ary) is actx

    actx2 = actx.clone()

    ary_dof_frozen = freeze(ary_dof)
    with pytest.raises(ValueError) as exc_info:
        ary_dof + ary_dof_frozen

    assert "frozen" in str(exc_info.value)

    ary_dof_2 = thaw(freeze(ary_dof), actx2)

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
    if np.prod(shapes) == 0:
        # https://github.com/inducer/loopy/pull/497
        # NOTE: only fails for the pytato array context at the moment
        pytest.xfail("strides do not match in subary")

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


def test_flatten_array_container_failure(actx_factory):
    actx = actx_factory()

    from arraycontext import flatten, unflatten
    ary = _get_test_containers(actx, shapes=512)[0]
    flat_ary = flatten(ary, actx)

    with pytest.raises(TypeError):
        # cannot unflatten from a numpy array
        unflatten(ary, actx.to_numpy(flat_ary), actx)

    with pytest.raises(ValueError):
        # cannot unflatten non-flat arrays
        unflatten(ary, flat_ary.reshape(2, -1), actx)

    with pytest.raises(ValueError):
        # cannot unflatten partially
        unflatten(ary, flat_ary[:-1], actx)

# }}}


# {{{ test from_numpy and to_numpy

def test_numpy_conversion(actx_factory):
    actx = actx_factory()

    nelements = 42
    ac = MyContainer(
            name="test_numpy_conversion",
            mass=np.random.rand(nelements, nelements),
            momentum=make_obj_array([np.random.rand(nelements) for _ in range(3)]),
            enthalpy=np.array(np.random.rand()),
            )

    from arraycontext import from_numpy, to_numpy
    ac_actx = from_numpy(ac, actx)
    ac_roundtrip = to_numpy(ac_actx, actx)

    assert np.allclose(ac.mass, ac_roundtrip.mass)
    assert np.allclose(ac.momentum[0], ac_roundtrip.momentum[0])

    from dataclasses import replace
    ac_with_cl = replace(ac, enthalpy=ac_actx.mass)
    with pytest.raises(TypeError):
        from_numpy(ac_with_cl, actx)

    with pytest.raises(TypeError):
        from_numpy(ac_actx, actx)

    with pytest.raises(TypeError):
        to_numpy(ac, actx)

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


def scale_and_orthogonalize(alpha, vel):
    from arraycontext import rec_map_array_container
    actx = vel.array_context
    scaled_vel = rec_map_array_container(lambda x: alpha * x,
                                         vel)
    return Velocity2D(-scaled_vel.v, scaled_vel.u, actx)


def test_actx_compile(actx_factory):
    from arraycontext import (to_numpy, from_numpy)
    actx = actx_factory()

    compiled_rhs = actx.compile(scale_and_orthogonalize)

    v_x = np.random.rand(10)
    v_y = np.random.rand(10)

    vel = from_numpy(Velocity2D(v_x, v_y, actx), actx)

    scaled_speed = compiled_rhs(np.float64(3.14), vel)

    result = to_numpy(scaled_speed, actx)
    np.testing.assert_allclose(result.u, -3.14*v_y)
    np.testing.assert_allclose(result.v, 3.14*v_x)


def test_actx_compile_python_scalar(actx_factory):
    from arraycontext import (to_numpy, from_numpy)
    actx = actx_factory()

    compiled_rhs = actx.compile(scale_and_orthogonalize)

    v_x = np.random.rand(10)
    v_y = np.random.rand(10)

    vel = from_numpy(Velocity2D(v_x, v_y, actx), actx)

    scaled_speed = compiled_rhs(3.14, vel)

    result = to_numpy(scaled_speed, actx)
    np.testing.assert_allclose(result.u, -3.14*v_y)
    np.testing.assert_allclose(result.v, 3.14*v_x)


def test_actx_compile_kwargs(actx_factory):
    from arraycontext import (to_numpy, from_numpy)
    actx = actx_factory()

    compiled_rhs = actx.compile(scale_and_orthogonalize)

    v_x = np.random.rand(10)
    v_y = np.random.rand(10)

    vel = from_numpy(Velocity2D(v_x, v_y, actx), actx)

    scaled_speed = compiled_rhs(3.14, vel=vel)

    result = to_numpy(scaled_speed, actx)
    np.testing.assert_allclose(result.u, -3.14*v_y)
    np.testing.assert_allclose(result.v, 3.14*v_x)

# }}}


# {{{ test_container_equality

def test_container_equality(actx_factory):
    actx = actx_factory()

    ary_dof, _, _, dc_of_dofs, bcast_dc_of_dofs = \
            _get_test_containers(actx)
    _, _, _, dc_of_dofs_2, bcast_dc_of_dofs_2 = \
            _get_test_containers(actx)

    # MyContainer sets eq_comparison to False, so equality comparison should
    # not succeed.
    dc = MyContainer(name="yoink", mass=ary_dof, momentum=None, enthalpy=None)
    dc2 = MyContainer(name="yoink", mass=ary_dof, momentum=None, enthalpy=None)
    assert dc != dc2

    assert isinstance(bcast_dc_of_dofs == bcast_dc_of_dofs_2, MyContainerDOFBcast)

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

    @property
    def array_context(self):
        return self.u.array_context


def test_leaf_array_type_broadcasting(actx_factory):
    # test support for https://github.com/inducer/arraycontext/issues/49
    actx = actx_factory()

    foo = Foo(DOFArray(actx, (actx.zeros(3, dtype=np.float64) + 41, )))
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
