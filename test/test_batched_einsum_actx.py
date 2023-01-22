import pytest


try:
    import feinsum  # noqa: F401
except ModuleNotFoundError:
    pytest.skip(reason="BatchedEinsumActx imposes feinsum as a hard dep.",
                allow_module_level=True)

try:
    from loopy import get_kennedy_unweighted_fusion_candidates  # noqa: F401
    from loopy import rename_inames_in_batch  # noqa: F401
except ImportError:
    pytest.skip(reason="BatchedEinsumActx imposes loop-fusion support in "
                "loopy as a hard dep.", allow_module_level=True)

from dataclasses import dataclass

import numpy as np

from pytools.tag import UniqueTag

from arraycontext import (
    BatchedEinsumPytatoPyOpenCLArrayContext as BaseBatchedEinsumArrayContext,
    PyOpenCLArrayContext, PytatoPyOpenCLArrayContext, tag_axes)
from arraycontext.pytest import (
    _PytestEagerJaxArrayContextFactory, _PytestPyOpenCLArrayContextFactoryWithClass,
    _PytestPytatoJaxArrayContextFactory, _PytestPytatoPyOpenCLArrayContextFactory,
    _PytestSplitPytatoPyOpenCLArrayContextFactory,
    pytest_generate_tests_for_array_contexts)


# {{{ axes tag types for image processing

class AxisTagsForTesting(UniqueTag):
    pass


class ImageDimensionTag(AxisTagsForTesting):
    """
    An abstract tag type that is tagged to an array's axis indexing along an image's
    axis.
    """


class XDimension(ImageDimensionTag):
    """
    A tag that is attached to a :class:`pytato.array.Axis` that indexes along the
    x-dimension of an image.
    """


class YDimension(ImageDimensionTag):
    """
    A tag that is attached to a :class:`pytato.array.Axis` that indexes along the
    y-dimension of an image.
    """


class ChannelDimension(ImageDimensionTag):
    """
    A tag that is attached to a :class:`pytato.array.Axis` that indexes along the
    channels of an image.
    """

# }}}


# {{{ generic axes tags

@dataclass(frozen=True)
class NamedAxis(AxisTagsForTesting):
    name: str

# }}}


# {{{ array context fixture

class BatchedEinsumPytatoPyOpenCLArrayContext(
        BaseBatchedEinsumArrayContext):
    def __init__(self, queue, allocator=None):
        super().__init__(queue, allocator,
                         fallback_to_no_fusion=False,
                         loop_fusion_axis_tag_t=AxisTagsForTesting)


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


class _PytatoPyOpenCLArrayContextForTestsFactory(
        _PytestPytatoPyOpenCLArrayContextFactory):
    actx_class = _PytatoPyOpenCLArrayContextForTests


class _PyOpenCLArrayContextForTestsFactoryWithHostScalars(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    force_device_scalars = True
    actx_class = _PyOpenCLArrayContextForTests


class _PytestBatchedEinsumPytatoPyOpenCLArrayContextFactory(
        _PytestPytatoPyOpenCLArrayContextFactory):
    @property
    def actx_class(self):
        return BatchedEinsumPytatoPyOpenCLArrayContext


pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    _PyOpenCLArrayContextForTestsFactoryWithHostScalars,
    _PytatoPyOpenCLArrayContextForTestsFactory,
    _PytestEagerJaxArrayContextFactory,
    _PytestPytatoJaxArrayContextFactory,
    _PytestSplitPytatoPyOpenCLArrayContextFactory,
    _PytestBatchedEinsumPytatoPyOpenCLArrayContextFactory,
    ])

# }}}


def test_simple_add(actx_factory):
    # Lesson 01 of Halide Tutorial
    actx = actx_factory()

    rng = np.random.default_rng(0)
    a_np = rng.random((800, 600))
    b_np = rng.random((800, 600))
    a = actx.from_numpy(a_np)
    b = actx.from_numpy(b_np)

    a = tag_axes(actx, {0: XDimension(), 1: YDimension()}, a)
    b = tag_axes(actx, {0: XDimension(), 1: YDimension()}, b)

    out = actx.to_numpy(a + b)
    ref_out = a_np + b_np

    np.testing.assert_allclose(out, ref_out)


def test_brighten_image(actx_factory):
    # Lesson 02 of Halide Tutorial
    actx = actx_factory()

    rng = np.random.default_rng(0)

    img_np = 255*rng.random((800, 600, 3), dtype=np.float32)

    img = actx.from_numpy(img_np)
    img = tag_axes(actx,
                   {0: XDimension(), 1: YDimension(), 2: ChannelDimension()},
                   img)

    brightened_img = 1.5*img
    clamped_brightened_img = actx.np.minimum(brightened_img, np.float32(255))

    out = actx.to_numpy(clamped_brightened_img)
    ref_out = np.minimum(1.5*img_np, np.float32(255))

    np.testing.assert_allclose(out, ref_out)


def test_simple_einsum(actx_factory):
    actx = actx_factory()

    rng = np.random.default_rng()

    a_np = rng.random((10, 4))
    a = actx.from_numpy(a_np)
    a = tag_axes(actx,
                 {0: XDimension(), 1: YDimension()}, a)

    out1 = actx.einsum("ij,ij->i", a, a+1)
    out2 = actx.einsum("ij,ij->i", 2*a, 3*a+7)

    ref_out = (np.einsum("ij,ij->i", a_np, a_np + 1)
               + np.einsum("ij,ij->i", 2*a_np, 3*a_np+7))
    out = actx.to_numpy(out1 + out2)

    np.testing.assert_allclose(ref_out, out)


def test_nested_einsum(actx_factory):
    actx = actx_factory()

    rng = np.random.default_rng()

    a_np = rng.random((10, 4))

    # {{{ compute out

    a = actx.from_numpy(a_np)
    a = tag_axes(actx,
                 {0: XDimension(), 1: YDimension()}, a)
    b = a + 1

    out1 = actx.einsum("ij,ij->i", a, b)
    out2 = actx.einsum("ij,ij->i", 2*a, 3*a+7)
    out3 = actx.einsum("ij,i->i", 3*b, 2*out1)

    out = actx.to_numpy(out1 + out2 + out3)

    # }}}

    # {{{ compute ref_out

    b_np = a_np + 1
    out1_np = np.einsum("ij,ij->i", a_np, a_np+1)
    out2_np = np.einsum("ij,ij->i", 2*a_np, 3*a_np+7)
    out3_np = np.einsum("ij,i->i", 3*b_np, 2*out1_np)
    ref_out = out1_np + out2_np + out3_np

    # }}}

    np.testing.assert_allclose(ref_out, out)


def test_dg_3d_divergence(actx_factory):
    actx = actx_factory()
    rng = np.random.default_rng(42)
    n_el = 1000
    n_dof = 35

    ax_np, ay_np, az_np = rng.random((3, n_el, n_dof))
    jac_np = rng.random((3, 3, n_el))
    mat_np = rng.random((3, n_dof, n_dof))

    ax, ay, az = (actx.from_numpy(ax_np),
                  actx.from_numpy(ay_np),
                  actx.from_numpy(az_np))
    jac = actx.from_numpy(jac_np)
    jac = tag_axes(actx, {0: NamedAxis("x"),
                          1: NamedAxis("r"),
                          2: NamedAxis("e")}, jac)
    mat = actx.from_numpy(mat_np)
    mat = tag_axes(actx, {0: NamedAxis("r"),
                          1: NamedAxis("i"),
                          2: NamedAxis("j")}, mat)

    out = 2*actx.einsum(
        "xre,rij,xej->ei",
        jac, mat, actx.np.stack([3*actx.np.sin(ax) + 4*actx.np.cos(ax),
                                 12*actx.np.exp(ay) + 5*actx.np.cos(ay),
                                 8*az]))
    ref_out = 2*np.einsum(
        "xre,rij,xej->ei",
        jac_np, mat_np, np.stack([3*np.sin(ax_np) + 4*np.cos(ax_np),
                                  12*np.exp(ay_np) + 5*np.cos(ay_np),
                                  8*az_np]))

    np.testing.assert_allclose(ref_out, actx.to_numpy(out))

# vim: fdm=marker
