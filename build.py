from __future__ import annotations

import os
import shutil
from pathlib import Path

from Cython.Build import build_ext, cythonize
from setuptools import Distribution, Extension

# import Cython.Compiler.Options
# Cython.Compiler.Options.cimport_from_pyx = True

SOURCE_DIR = Path("koerce")
BUILD_DIR = Path("cython_build")


cythonized_modules = cythonize(
    [
        Extension(
            "koerce.builders",
            ["koerce/builders.py"],
        ),
        Extension(
            "koerce.patterns",
            ["koerce/patterns.py"],
        ),
    ],
    build_dir=BUILD_DIR,
    # generate anannotated .html output files.
    annotate=True,
    # nthreads=multiprocessing.cpu_count() * 2,
    compiler_directives={
        "language_level": "3",
        "binding": False,
        "boundscheck": False,
        "nonecheck": False,
        "always_allow_keywords": False,
        # "wraparound": False,
        # "profile": True,
        # "annotation_typing": False
    },
    # always rebuild, even if files untouched
    force=True,
    # emit_linenums=True
)

dist = Distribution({"ext_modules": cythonized_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

# remove all binaries *.so, *.dll, *.lib, *.pyd, *.dylib
for ext in ["so", "dll", "lib", "pyd", "dylib"]:
    for f in SOURCE_DIR.glob(f"*.{ext}"):
        print(f"Removing previously built binary {f}")  # noqa: T201
        f.unlink()

for output in cmd.get_outputs():
    relative_extension = os.path.relpath(output, cmd.build_lib)
    print(f"Copying {output} to {relative_extension}")  # noqa: T201
    shutil.copyfile(output, relative_extension)
