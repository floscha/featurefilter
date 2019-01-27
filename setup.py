from setuptools import setup


setup(
    name='featurefilter',
    version='0.0.1',
    description='A Python library for removing uninformative variables from datasets',
    url='https://github.com/floscha/featurefilter/',
    author='Florian Schäfer',
    author_email='florian.joh.schaefer@gmail.com',
    license='MIT',
    packages=['featurefilter'],
    install_requires=[
        'pandas'
    ],
    zip_safe=False
)
