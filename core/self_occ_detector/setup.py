from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='selfoccdtc_ops',
    ext_modules=[
        CUDAExtension('selfoccdtc_ops', [
            'selfoccdtc_cpp.cpp',
            'selfoccdtc_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })