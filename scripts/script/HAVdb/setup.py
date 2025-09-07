from setuptools import setup, find_packages

setup(
    name='HAVdb',
    version='0.1',
    description='Python library for importing and segmenting data',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn'
    ],
)