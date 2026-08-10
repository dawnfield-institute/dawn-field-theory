from setuptools import setup, find_packages

setup(
    name='navier_symbolic_engine',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'torch>=1.9.0',
        'numba>=0.54.0',
        'h5py>=3.3.0',
        'pyyaml>=5.4.0',
    ],
    author='Peter Groom',
    description='Navier-Stokes Symbolic Collapse Framework',
    license='MIT',
)
