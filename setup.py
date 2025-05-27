from setuptools import setup, Extension
import sys


smolgrad_module = Extension(
    "SmolGrad.tensor",
    sources=["src/tensor.c"],
    include_dirs=["include"],
    extra_compile_args=["-O3"] if sys.platform != "win32" else ["/O2"],
)

setup(
    name="SmolGrad",
    version="0.1.0",
    description="A smol automatic differentiation library",
    packages=["SmolGrad"],
    ext_modules=[smolgrad_module],
    python_requires=">=3.7"
)
