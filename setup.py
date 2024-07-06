# from aimnet2_tscode.__main__ import __version__
from setuptools import setup, find_packages

long_description = ("Python bindings to use Anstine, Zubatyuk and Isayev\'s AIMNET2 Neural Network model via an ASE Calculator for FIRECODE.")

# with open('CHANGELOG.md', 'r') as f:
#     long_description += '\n\n'
#     long_description += f.read()

setup(
    name='aimnet2_firecode',
    version="1.0.0",
    # description='Computational chemistry general purpose transition state builder and ensemble optimizer',
    # keywords=['computational chemistry', 'ASE', 'transition state', 'xtb'],

    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: OS Independent',
    ],

    long_description=long_description,
    long_description_content_type='text/markdown',

    install_requires=[
        'ase',
        'torch',
        'firecode',
    ],

    url='https://www.github.com/ntampellini/aimnet2_firecode',
    author='Nicolò Tampellini',
    author_email='nicolo.tampellini@yale.edu',

    packages=find_packages(),
    python_requires=">=3.8",
)