"""Portable pybind11 build: python -m pip install --no-build-isolation ."""
import os

from setuptools import Extension, setup
import pybind11

setup(
    name="hunyuan-mesh-inpaint",
    version="0.1.0",
    ext_modules=[Extension(
        "mesh_inpaint_processor",
        ["mesh_inpaint_processor.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"],
    )],
)
