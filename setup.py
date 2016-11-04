from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += [
        Extension("utils.cutils", [ "utils/cutils.pyx" ], extra_compile_args=["-O3"] ),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("utils.cutils", [ "utils/cutils.c" ]),
    ]

setup(
    name='utils',
    cmdclass = cmdclass,
    ext_modules=ext_modules,
)
