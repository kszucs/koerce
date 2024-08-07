from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# setuptools *must* come before Cython, otherwise Cython's distutils hacking
# will override setuptools' build_ext command and cause problems in other build
# systems such as nix.

from setuptools import Distribution, Extension
from Cython.Build import build_ext, cythonize

SOURCE_DIR = Path("koerce")
BUILD_DIR = Path("cython_build")


extensions = [
    Extension("koerce.annots", ["koerce/annots.py"]),
    Extension("koerce.builders", ["koerce/builders.py"]),
    Extension("koerce.patterns", ["koerce/patterns.py"]),
    # Extension("koerce.utils", ["koerce/utils.py"]),
]

cythonized_modules = cythonize(
    extensions,
    build_dir=BUILD_DIR,
    # generate anannotated .html output files.
    annotate=True,
    compiler_directives={
        "language_level": "3",
        "binding": False,
        "boundscheck": False,
        "nonecheck": False,
        "always_allow_keywords": False,
    },
)

dist = Distribution({"ext_modules": cythonized_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

# remove all binaries *.so, *.dll, *.lib, *.pyd, *.dylib
for ext in ["so", "dll", "lib", "pyd", "dylib"]:
    for f in SOURCE_DIR.glob(f"*.{ext}"):
        f.unlink()
        print(f"removed {f!r}", file=sys.stderr)  # noqa: T201

for output in cmd.get_outputs():
    relative_extension = os.path.relpath(output, cmd.build_lib)
    print(f"{output} -> {relative_extension}", file=sys.stderr)  # noqa: T201
    shutil.copyfile(output, relative_extension)
