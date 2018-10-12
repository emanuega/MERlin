# MERFISH Analysis Code

This repository contains code for analyzing and decoding MERFISH images.

## Getting Started

# Data format specifications

## Raw images

## Data organization

## Codebook

## Position list

A csv file containing a list of positions is expected either in the raw data directory or the parent directory. Alternatively, the name of a position csv file that is located in POSITION\_HOME can be provided.

 This text file should contain 'positions' in the file name. The i'th row in the file should be coordinates of the i'th field of view. Each row should contain the x position and the y position with a comma in between.

### Prerequisites

A .env file is required to specify the search locations for the various input and output files. The following variables should be defined in a file named .env in the project root directory:

- DATA\_HOME - The path of the root directory to the raw data.
- ANALYSIS\_HOME - The path of the root directory where analysis results should be stored.
- CODEBOOK\_HOME - The path of the directory where codebooks can be found.
- DATA\_ORGANIZATION\_HOME - The path of the directory where data organization files can be found.
- POSITION\_HOME - The path of the directory where position files can be found.
- PARAMETERS\_HOME - The path to the directory where analysis parameters can be found.

The contents of an example .env file are below:

```python
DATA_HOME=D:/data
ANALYSIS_HOME=D:/analysis
CODEBOOK_HOME=D:/merfish-parameters/codebooks
DATA_ORGANIZATION_HOME=D:/merfish-parameters/dataorganization
POSITION_HOME=D:/merfish-parameters/positions
PARAMETERS_HOME=D:/merfish-paramters/analysis_parameters
```

### Installing

This module requires python 3.6 and above. [Storm-analysis](https://github.com/ZhuangLab/storm-analysis) must be intalled prior to installing this package. Additionally, the package rtree is not properly installed by pip and should be installed independently. For example, using Anaconda:

```
conda install rtree
```

This module can be install with pip:

```
pip install --process-dependency-links -e MERlin
```

## Usage

After installation, MERlin can be run from the command line with the input parameters specified, such as: 

```
merlin -d test_data -a test_analysis_parameters.json -o Culture_16bits -c HAEC1E1 -n 5
```
Here the images are contained in the directory %DATA_HOME%\test_data\ and the analysis tasks listed in test_analysis_parameters.json are run with data organization Culture_16bits and codebook HAEC1E1 using 5 cores for each process. 
## Running the tests

## Built With

## Authors

* **George Emanuel** - *Initial work* 

