"""Setup script for excitation-rk4-sparse package."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    requirements = [
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
    ]
    return requirements

setup(
    name='rk4-sparse-cpp',
    version='0.2.0',
    author='Hiroki Tsusaka',
    author_email='tsusaka4research@gmail.com',
    description='High-performance sparse matrix RK4 solver for quantum excitation dynamics',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/1160-hrk/rk4-sparse-cpp',
    project_urls={
        'Bug Reports': 'https://github.com/1160-hrk/rk4-sparse-cpp/issues',
        'Source': 'https://github.com/1160-hrk/rk4-sparse-cpp',
        'Documentation': 'https://github.com/1160-hrk/rk4-sparse-cpp/tree/main/docs',
    },
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    keywords='quantum dynamics rk4 sparse physics simulation',
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'flake8>=4.0',
        ],
        'benchmark': [
            'memory-profiler>=0.58',
            'line-profiler>=3.0',
            'jupyter>=1.0',
            'seaborn>=0.11',
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 