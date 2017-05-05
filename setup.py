from setuptools import setup, find_packages

setup(
    name='sparsedb',
    version='0.1dev',
    packages=find_packages(),
    license='BSD 2-clause "Simplified" License',
    install_requires=[
        "h5py>=2.7.0",
        "pyroaring>=0.1.1",
        "pytoml>=0.1.12",
        "scipy",
    ],
)
