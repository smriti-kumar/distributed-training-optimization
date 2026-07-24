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

setup(
    name='quiptools_quant',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'quiptools_quant',
            ['quiptools_quant_wrapper.cpp', 'quiptools_quant.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '-std=c++17', '-lineinfo'],
            })
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
