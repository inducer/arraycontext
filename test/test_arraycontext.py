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
        FirstAxisIsElementsTag)
from arraycontext import (  # noqa: F401
        pytest_generate_tests_for_pyopencl_array_context
        as pytest_generate_tests,
        _acf)

import logging
logger = logging.getLogger(__name__)


# {{{ stand-in DOFArray implementation

@with_container_arithmetic(
        bcast_obj_array=True,
        bcast_numpy_array=True,
        rel_comparison=True)
class DOFArray:
    def __init__(self, actx, data):
        if not (actx is None or isinstance(actx, ArrayContext)):
            raise TypeError("actx must be of type ArrayContext")

        if not isinstance(data, tuple):
            raise TypeError("'data' argument must be a tuple")

        self.array_context = actx
        self.data = data

    __array_priority__ = 10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

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
    return enumerate(ary.data)


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
    actx_result = actx.to_numpy(op(actx.np, *dofarray_args)[0])

    assert np.allclose(actx_result, ref_result)

    # }}}

    # {{{ test object arrays of DOFArrays

    obj_array_args = [
            make_obj_array([arg]) if isinstance(arg, DOFArray) else arg
            for arg in dofarray_args]
    obj_array_result = actx.to_numpy(op(actx.np, *obj_array_args)[0][0])

    assert np.allclose(obj_array_result, ref_result)

    # }}}

# }}}


# {{{ np.function same as numpy

@pytest.mark.parametrize(("sym_name", "n_args"), [
            ("sin", 1),
            ("exp", 1),
            ("arctan2", 2),
            ("minimum", 2),
            ("maximum", 2),
            ("where", 3),
            ("conj", 1),
            ])
def test_array_context_np_workalike(actx_factory, sym_name, n_args):
    actx = actx_factory()

    ndofs = 5000
    args = [np.random.randn(ndofs) for i in range(n_args)]
    assert_close_to_numpy_in_containers(
            actx, lambda _np, *_args: getattr(_np, sym_name)(*_args), args)


@pytest.mark.parametrize(("sym_name", "n_args"), [
            # ("empty_like", 1),    # NOTE: fails np.allclose, obviously
            ("zeros_like", 1),
            ("ones_like", 1),
            ])
def test_array_context_np_like(actx_factory, sym_name, n_args):
    actx = actx_factory()

    ndofs = 5000
    args = [np.random.randn(ndofs) for i in range(n_args)]
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

# }}}


# {{{ arithmetic same as numpy

def test_dof_array_arithmetic_same_as_numpy(actx_factory):
    actx = actx_factory()

    ndofs = 50_000

    def get_real(ary):
        return ary.real

    def get_imag(ary):
        return ary.real

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

            actx_result = actx.to_numpy(op_func(*actx_args)[0])

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
                        op_func(*obj_array_args)[0][0])

                assert np.allclose(obj_array_result, ref_result)

            # }}}

# }}}


# {{{ reductions same as numpy

def test_dof_array_reductions_same_as_numpy(actx_factory):
    actx = actx_factory()

    from numbers import Number
    for name in ["sum", "min", "max"]:
        ary = np.random.randn(3000)
        np_red = getattr(np, name)(ary)
        actx_red = getattr(actx.np, name)(actx.from_numpy(ary))

        assert isinstance(actx_red, Number)
        assert np.allclose(np_red, actx_red)

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


# {{{ test array container

@with_container_arithmetic(bcast_obj_array=False, rel_comparison=True)
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


def _get_test_containers(actx, ambient_dim=2):
    x = DOFArray(actx, (actx.from_numpy(np.random.randn(50_000)),))

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    dataclass_of_dofs = MyContainer(
            name="container",
            mass=x,
            momentum=make_obj_array([x, x]),
            enthalpy=x)

    ary_dof = x
    ary_of_dofs = make_obj_array([x, x, x])
    mat_of_dofs = np.empty((2, 2), dtype=object)
    for i in np.ndindex(mat_of_dofs.shape):
        mat_of_dofs[i] = x

    return ary_dof, ary_of_dofs, mat_of_dofs, dataclass_of_dofs


def test_container_multimap(actx_factory):
    actx = actx_factory()
    ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs = _get_test_containers(actx)

    # {{{ check

    def _check_allclose(f, arg1, arg2, atol=1.0e-14):
        assert np.linalg.norm((f(arg1) - arg2).get()) < atol

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
    ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs = _get_test_containers(actx)

    # {{{ check

    def _check_allclose(f, arg1, arg2, atol=1.0e-14):
        assert np.linalg.norm((f(arg1) - arg2).get()) < atol

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

    # }}}


def test_container_freeze_thaw(actx_factory):
    actx = actx_factory()
    ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs = _get_test_containers(actx)

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

    # }}}


@pytest.mark.parametrize("ord", [2, np.inf])
def test_container_norm(actx_factory, ord):
    actx = actx_factory()

    ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs = _get_test_containers(actx)

    from pytools.obj_array import make_obj_array
    c = MyContainer(name="hey", mass=1, momentum=make_obj_array([2, 3]), enthalpy=5)
    n1 = actx.np.linalg.norm(make_obj_array([c, c]), ord)
    n2 = np.linalg.norm([1, 2, 3, 5]*2, ord)

    assert abs(n1 - n2) < 1e-12

# }}}


# {{{ test from_numpy and to_numpy

def test_numpy_conversion(actx_factory):
    actx = actx_factory()

    ac = MyContainer(
            name="test_numpy_conversion",
            mass=np.random.rand(42),
            momentum=make_obj_array([np.random.rand(42) for _ in range(3)]),
            enthalpy=np.random.rand(42),
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

    with pytest.raises(ValueError):
        to_numpy(ac, actx)

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
