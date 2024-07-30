from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import Distribution

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    print(
        "Cython is required to build the Cython modules. "
        "Please install Cython first by running: pip install Cython"
    )

# import Cython.Compiler.Options
# Cython.Compiler.Options.cimport_from_pyx = True

SOURCE_DIR = Path("koerce")
BUILD_DIR = Path("cython_build")


cythonized_modules = cythonize(
    [
        "koerce/patterns.py",
        "koerce/builders.py",
    ],
    build_dir=BUILD_DIR,
    # generate anannotated .html output files.
    annotate=True,
    # nthreads=multiprocessing.cpu_count() * 2,
    compiler_directives={"language_level": "3"},
    # always rebuild, even if files untouched
    force=False,
)

dist = Distribution({"ext_modules": cythonized_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

for output in cmd.get_outputs():
    relative_extension = os.path.relpath(output, cmd.build_lib)
    print(f"Copying {output} to {relative_extension}")
    shutil.copyfile(output, relative_extension)
