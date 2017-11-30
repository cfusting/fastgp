from setuptools import setup, find_packages

setup(
    name='fastgp',
    version='0.1.0',
    description='Fast genetic programming.',
    author='Chris Fusting',
    author_email='cfusting@gmail.com',
    license='GNU GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3'
    ],
    keywords='evolution machine learning artificial intelligence',
    install_requires=[
        'scipy',
        'deap',
        'cachetools',
        'numpy',
    ],
    python_requires='>=2.7',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    url='https://github.com/cfusting/fastgp'
)
