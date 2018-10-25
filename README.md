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

The MERlin analysis results can be explored using MERlinView. The analysis of the data set test_data can be explored with MERlinView from the command line as:

merlinview -d test_data

## Analysis tasks

### warp.FiducialFitWarp

Description: Aligns image stacks by fitting fiducial spots.

### warp.FiducialCorrelationWarp

Description: Aligns image stacks by maximizing the cross correlation between fiducial images. 

### preprocess.DecovolutionPreprocess

Description: High-pass filters and deconvolves the image data in peparation for bit calling.

Parameters:
* warp_task - The name of the warp task that provides the aligned image stacks.
* highpass_pass - The standard deviation to use for the high pass filter.
* decon_sigma - The standard deviation to use for the lucy richardson deconvolution.

### optimize.Optimize

Description: Determines the optimal per-bit scale factors for barcode decoding.

Parameters:
* iteration_count - The number of iterations to perform for the optimization.
* fov_per_iteration - The number of fields of view to decode in each round of optimization.

### decode.Decode

Description: Extract barcodes from all field of views.

### filterbarcodes.FilterBarcodes

Description: Filters the decoded barcodes based on area and intensity

Parameters:
* area_threshold - Barcodes with areas below the specified area_threshold are
removed.
* intensity_threshold - Barcodes with intensities below this threshold are removed.

### segment.SegmentCells

Description: Determines cell boundaries using a nuclear stain and a cell stain.

Parameters:
* nucleus_threshold - The relative intensity threshold for seeding the nuclei.
* cell_threshold - The relative intensity threshold for setting boundaries for the watershed.
* nucleus_index - The image index for the nucleus image.
* cell_index - The image index for the cell image.

### segment.CleanCellSegmentation

Description: Cleans the cell segmentation by merging cells that were fargmented along the field of view boundaries.

### generatemosaic.GenerateMosaic

Description: Assembles the images from each field of view into a low resolution mosaic.

Parameters:
* microns_per_pixel - The number of microns to correspond with a pixel in the mosaic.

## Running the tests

## Built With

## Authors

* **George Emanuel** - *Initial work* 

