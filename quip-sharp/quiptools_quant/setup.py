import os

from setuptools import setup
from torch.utils import cpp_extension

# Modeled on ../quiptools/setup.py. -arch is left to the extension's default
# (nvcc will target the architecture of the GPU(s) visible at build time via
# TORCH_CUDA_ARCH_LIST / the environment torch.utils.cpp_extension detects);
# set TORCH_CUDA_ARCH_LIST explicitly if building on a machine without the
# target GPU present. --use_fast_math is intentionally NOT passed: it enables
# lower-precision transcendentals/reciprocals that would break the bit-exact
# determinism this extension requires (workload_analysis.md section 5).
#
# The importable module name is `quiptools_quant_cuda`, deliberately NOT
# `quiptools_quant` (this directory's name) -- same reason ../quiptools/
# builds a module called `quiptools_cuda` rather than `quiptools`. An
# extension whose Python import name matches its own source directory name
# can resolve to the bare source directory (an empty PEP 420 namespace
# package, since there's no __init__.py here) instead of the compiled .so
# once the repo root ends up on sys.path -- which it usually does (pytest's
# rootdir insertion, scripts run from the repo root, etc). That failure mode
# looks exactly like "import succeeds but the module has none of the bound
# functions" -- if you ever see that, this is the first thing to check.

setup(
    name='quiptools_quant_cuda',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'quiptools_quant_cuda',
            ['quiptools_quant_wrapper.cpp', 'quiptools_quant.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                # --expt-relaxed-constexpr: Kernel 3 (quiptools_quant.cu Section E)
                # uses constexpr helper functions with loops (subtree_level_off etc)
                # from __device__/__global__ template code to compute shared-memory
                # layouts at compile time; nvcc needs this flag to allow that.
                'nvcc': ['-O3', '-std=c++17', '-lineinfo', '--expt-relaxed-constexpr'],
            })
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
