from setuptools import setup, find_packages

setup(
    name='ag-learning',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'numpy<2.0',
        'pandas>=2.1.2',
        'scipy>=1.9.1',
        'scikit-learn>=1.0.2',
        'scikit-learn-intelex==2021.20220207.124119',
        'pandas>=2.1.2',
        'matplotlib>=3.1.0',
        'GDAL>=3.5.1',
        'cvxopt>=1.3.0'
    ],
    author='Willian Oliveira',
    description='A multilevel agglomerative learning strategy for remote sensing image classification',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wivoliveira/ag-learning',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Apache License 2.0',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
