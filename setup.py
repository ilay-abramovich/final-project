from setuptools import setup, Extension
import numpy

ext = Extension(
    "symnmfmodule",
    sources=["symnmfmodule.c", "symnmf.c"],
    include_dirs=[numpy.get_include(), "/usr/include/"],
)

setup(name="symnmfmodule", version="1.0", ext_modules=[ext])
