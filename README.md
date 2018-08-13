# MERFISH Analysis Code

This repository contains code for analyzing and decoding MERFISH images.

## Getting Started

To set the default analysis directories, the following variables should be defined in a file named .env in the project root directory:

- DATA\_HOME - The path of the root directory to the raw data.
- ANALYSIS\_HOME - The path of the root directory where analysis results should be stored.
- CODEBOOK\_HOME - The path of the directory where codebooks can be found.
- DATA\_ORGANIZATION\_HOME - The path of the directory where data organization files can be found.
- POSITION\_HOME - The path of the directory where position files can be found.

An example .env file is below:


```python
DATA_HOME=//10.245.74.90/data/flow_differentiation
ANALYSIS_HOME=D:/analysis
CODEBOOK_HOME=D:/merfish-parameters/codebooks
DATA_ORGANIZATION_HOME=D:/merfish-parameters/dataorganization
POSITION_HOME=D:/merfish-parameters/positions
```

### Prerequisites


### Installing

## Running the tests


## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **George Emanuel** - *Initial work* 

