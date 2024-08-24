from __future__ import annotations

import os
import shutil
import sys
import ast
from pathlib import Path

# setuptools *must* come before Cython, otherwise Cython's distutils hacking
# will override setuptools' build_ext command and cause problems in other build
# systems such as nix.

from setuptools import Distribution, Extension
from Cython.Build import build_ext, cythonize

SOURCE_DIR = Path("koerce")
BUILD_DIR = Path("cython_build")


def extract_imports_and_code(path):
    """Extracts the import statements and other code from python source."""
    with path.open("r") as file:
        tree = ast.parse(file.read(), filename=path.name)

    code = []
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        else:
            code.append(node)

    return imports, code


def ignore_import(imp, modules):
    absolute_names = ["koerce.{name}" for name in modules]
    if isinstance(imp, ast.ImportFrom):
        return imp.module in modules
    elif isinstance(imp, ast.Import):
        return imp.names[0].name in absolute_names
    else:
        raise TypeError(imp)


def concatenate_files(header, inputs, output):
    all_imports = []
    all_code = []
    modules = []

    for file_path in inputs:
        path = Path(SOURCE_DIR / file_path)
        imports, code = extract_imports_and_code(path)
        all_imports.extend(imports)
        all_code.extend(code)
        modules.append(path.stem)

    # Deduplicate imports by their unparsed code
    unique_imports = {ast.unparse(stmt): stmt for stmt in all_imports}

    # Write to the output file
    with (SOURCE_DIR / output).open("w") as out:
        # Write the header
        for line in header:
            out.write(line)
            out.write("\n")

        # Write unique imports
        for code, stmt in unique_imports.items():
            if not ignore_import(stmt, modules):
                out.write(code)
                out.write("\n")

        # Write the rest of the code
        for stmt in all_code:
            out.write(ast.unparse(stmt))
            out.write("\n\n\n")


concatenate_files(
    header=[
        "# cython: language_level=3, binding=False, boundscheck=False, nonecheck=False, always_allow_keywords=False"
    ],
    inputs=["builders.py", "patterns.py", "annots.py"],
    output="_internal.pyx",
)
extension = Extension("koerce._internal", ["koerce/_internal.pyx"])

cythonized_modules = cythonize(
    [extension],
    build_dir=BUILD_DIR,
    cache=True,
    show_all_warnings=False,
    annotate=True,
    # The directives below don't seem to work with cythonize
    # compiler_directives={
    #     "language_level": "3",
    #     "binding": False,
    #     "boundscheck": False,
    #     "nonecheck": False,
    #     "always_allow_keywords": False,
    # },
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
