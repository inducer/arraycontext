from urllib.request import urlopen

_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2021, University of Illinois Board of Trustees"
author = "Arraycontext Contributors"

ver_dic = {}
exec(compile(open("../arraycontext/version.py").read(), "../arraycontext/version.py",
    "exec"), ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
release = ver_dic["VERSION_TEXT"]

intersphinx_mapping = {
    "https://docs.python.org/3/": None,
    "https://numpy.org/doc/stable/": None,
    "https://documen.tician.de/pytools": None,
    "https://documen.tician.de/pymbolic": None,
    "https://documen.tician.de/pyopencl": None,
    "https://documen.tician.de/pytato": None,
    "https://documen.tician.de/loopy": None,
    "https://documen.tician.de/meshmode": None,
    "https://docs.pytest.org/en/latest/": None,
    "https://jax.readthedocs.io/en/latest/": None,
}

import sys
sys.ARRAYCONTEXT_BUILDING_SPHINX_DOCS = True
