import os
import shutil
import multiprocessing
from pathlib import Path
from setuptools import Distribution

from Cython.Build import cythonize
from Cython.Distutils import build_ext

SOURCE_DIR = Path("koerce")
BUILD_DIR = Path("cython_build")


# extension_modules = get_extension_modules()
cythonized_modules = cythonize(
    [
        # "koerce/utils.py",
        "koerce/patterns.py",
        "koerce/builders.py",
    ],
    # module_list=extension_modules,
    # Don't build in source tree (this leaves behind .c files)
    build_dir=BUILD_DIR,
    # Don't generate an .html output file. Would contain source.
    annotate=True,
    # Parallelize our build
    # nthreads=multiprocessing.cpu_count() * 2,
    # Tell Cython we're using Python 3. Becomes default in Cython 3
    compiler_directives={"language_level": "3"},
    # (Optional) Always rebuild, even if files untouched
    # force=True,
)

dist = Distribution({"ext_modules": cythonized_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

for output in cmd.get_outputs():
    relative_extension = os.path.relpath(output, cmd.build_lib)
    shutil.copyfile(output, relative_extension)
