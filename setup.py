from setuptools import setup


setup(
    name='featurefilter',
    version='0.20',
    description='A Python library for removing uninformative variables from datasets',
    url='https://github.com/floscha/featurefilter/',
    author='Florian Schäfer',
    author_email='florian.joh.schaefer@gmail.com',
    license='MIT',
    packages=['featurefilter'],
    install_requires=[
        'pandas',
        'scikit-learn'
    ],
    extras_require={
        'spark': ['pyspark>=2.4.0',
                  'koalas==0.9.0']
    },
    zip_safe=False
)
