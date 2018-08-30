# MERFISH Analysis Code

This repository contains code for analyzing and decoding MERFISH images.

## Getting Started

To set the default analysis directories, the following variables should be defined in a file named .env in the project root directory:

- DATA\_HOME - The path of the root directory to the raw data.
- ANALYSIS\_HOME - The path of the root directory where analysis results should be stored.
- CODEBOOK\_HOME - The path of the directory where codebooks can be found.
- DATA\_ORGANIZATION\_HOME - The path of the directory where data organization files can be found.
- POSITION\_HOME - The path of the directory where position files can be found.
- PARAMETERS\_HOME - The path to the directory where analysis parameters can be found.

The contents of an example .env file are below:


```python
DATA_HOME=//10.245.74.90/data/flow_differentiation
ANALYSIS_HOME=D:/analysis
CODEBOOK_HOME=D:/merfish-parameters/codebooks
DATA_ORGANIZATION_HOME=D:/merfish-parameters/dataorganization
POSITION_HOME=D:/merfish-parameters/positions
```

# Data format specifications

## Raw images

## Data organization

## Codebook

## Position list

A csv file containing a list of positions is expected either in the raw data directory or the parent directory. Alternatively, the name of a position csv file that is located in POSITION\_HOME can be provided.

 This text file should contain 'positions' in the file name. The i'th row in the file should be coordinates of the i'th field of view. Each row should contain the x position and the y position with a comma in between.

### Prerequisites

The dependent package rtree is not properly installed by pip. It should be installed using:

```
conda install rtree
```

### Installing

This module requires python 3.6 and above. Additionally, [storm-analysis](https://github.com/ZhuangLab/storm-analysis) must be intalled prior to installing this  package.

Before installation, a virtual environment can be specified:

```
python -m venv .venv
```

This module can be install with pip:

```
pip install --process-dependency-links -e merlin
```

## Usage

## Running the tests


## Built With

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **George Emanuel** - *Initial work* 

