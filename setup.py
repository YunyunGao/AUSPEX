import os
import sys
from setuptools import setup
from setuptools.command import install

setup(
    name='auspex',
    version='1.3.0',
    author='Yunyun Gao',
    author_email='yunyun.gao@desy.de',
    packages=['auspex', ],
    package_data={'auspex': ['lib/int_lib.so'],
    },
    include_package_data=True,
    license='LICENSE.txt',
    #long_description=long_description,
    #long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': ['auspex = auspex.__init__:run', ]
    }
)




