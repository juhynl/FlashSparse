from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='flashsparse',
    version='0.1',
    packages=find_packages(),
    cmdclass={
        'build_ext': BuildExtension
    }
)
