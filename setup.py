"""
this setup will check for dependencies and install phy_utils on your computer
"""
from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='phy_utils',
    version='0.0.2',
    url='https://github.com/phydev/phy_utils.git',
    author='Mauricio Moreira',
    author_email='phydev@protonmail.com',
    description='Personal collection of scripts for data anlysis.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['data analysis', 'datavis'],
    license='GNU GPLv3',
    platform='Python 3.7',
    packages=find_packages(),
    install_requires=['numpy >= 1.14.3',
                      'scipy >= 1.7.1',
                      'seaborn >= 0.12.0',
                      'pandas >= 1.5.3',
                      'matplotlib >= 3.7.1'],
)
