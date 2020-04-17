import os
import setuptools

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS :: MacOS X",
    "License  :: Restricted use",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]

install_requires = [line.rstrip() for line in open(
    os.path.join(os.path.dirname(__file__), "requirements.txt"))]

setuptools.setup(
    name="merlin",
    version="0.1.6",
    description="MERFISH decoding software",
    author="George Emanuel",
    author_email="emanuega0@gmail.com",
    license="Restricted use",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': ["merlin=merlin.merlin:merlin"]
    },
    classifiers=CLASSIFIERS
)
