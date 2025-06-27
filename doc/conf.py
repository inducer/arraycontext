from importlib import metadata
from urllib.request import urlopen


_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2021, University of Illinois Board of Trustees"
author = "Arraycontext Contributors"
release = metadata.version("arraycontext")
version = ".".join(release.split(".")[:2])

intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "loopy": ("https://documen.tician.de/loopy", None),
    "meshmode": ("https://documen.tician.de/meshmode", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pymbolic": ("https://documen.tician.de/pymbolic", None),
    "pyopencl": ("https://documen.tician.de/pyopencl", None),
    "pytato": ("https://documen.tician.de/pytato", None),
    "pytest": ("https://docs.pytest.org/en/latest/", None),
    "python": ("https://docs.python.org/3/", None),
    "pytools": ("https://documen.tician.de/pytools", None),
}

# Some modules need to import things just so that sphinx can resolve symbols in
# type annotations. Often, we do not want these imports (e.g. of PyOpenCL) when
# in normal use (because they would introduce unintended side effects or hard
# dependencies). This flag exists so that these imports only occur during doc
# build. Since sphinx appears to resolve type hints lexically (as it should),
# this needs to be cross-module (since, e.g. an inherited arraycontext
# docstring can be read by sphinx when building meshmode, a dependent package),
# this needs a setting of the same name across all packages involved, that's
# why this name is as global-sounding as it is.
import sys


sys._BUILDING_SPHINX_DOCS = True


nitpick_ignore_regex = [
    ["py:class", r"arraycontext\.context\.ContainerOrScalarT"],
    ["py:class", r"ArrayOrContainer"],
    ["py:class", r"ArrayOrScalar"],
    ["py:class", r"arraycontext.container._UserDefinedArithArrayContainer"],
    ["py:class", r"np.integer"],
    ]
