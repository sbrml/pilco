from distutils.util import convert_path
from setuptools import setup, find_packages


def readme() -> str:
    with open('README.md') as f:
        return f.read()


version_dict = {}

with open(convert_path('pilco/version.py')) as file:
    exec(file.read(), version_dict)

setup(
    name='sbrml-pilco',
    version=version_dict['__version__'],
    description='PILCO in TF2',
    long_description=readme(),
    classifiers=['Programming Language :: Python :: 3.6'],
    author='SBRML',
    author_email='gf332@cam.ac.uk, stratismar@gmail.com',
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'matplotlib',
        'tensorflow',
        'tqdm',
        'not-tf-opt',
    ],
    zip_safe=False,
)