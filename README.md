# MERFISH Analysis Code

This repository contains code for decoding and visualizing MERFISH datasets.

## Getting Started

# Data format specifications

## Raw images

## Data organization

The data organization file specifies which the images correspond to each readout. The data organization file is a csv file where each row designates one readout with a one row header. The information provided for each readout indicates where to find the corresponding images in the raw data and how to align the images between rounds.

The columns in the data organization file are:

- bitName - The name of the readout.
- imageType - The base name for the image file that contains the images for this readout, for example ```Conventional_750_650_561_488_405```
- imageRegExp - A regular expression specifying how image file names are constructed for each field of view and each imaging round. The parameters used in the regular expression are imageType, fov, and imagingRound, for example: ```(?P<imageType>[\w|-]+)_(?P<fov>[0-9]+)_(?P<imagingRound>[0-9]+)```. Here, ```imageType``` specifies the string indicated in the ```imageType``` column for the corresponding row, ```imagingRound``` specifies the designated ```imagingRound``` for the corresponding row, and ```fov``` is filled with all field of view indexes in the data set. 
- bitNumber - The bit number corresponding to this readout.
- imagingRound - The round of imaging where this readout is measured, starting from zero.
- color - The illumination color that is used to measure this readout.
- frame - The zero indexed frame or frames in the image file where images corresponding to this readout can be found. For a single frame, a single integer can be provided. For multiple frames, the frames can be provided as a list as ```[0, 1, 2, 3, 4, 5, 6]```
- zPos - The z position for each of the frames specified in the previous column. For only a single frame, the z position should be provided as a decimal number while for multiple frames a list should be provided as for frame.
- fiducialImageType - The base name for the image file that contains the fiducial images for aligning images this readout, for example ```Conventional_750_650_561_488_405```
- fiducialRegExp - A regular expression specifying how file names are constructed for the fiducial image files. This regex follows the same format as ```imageRegExp```.
- fiducialImagingRound - The imaging round (zero indexed) corresponding to the fiducial images for aligning images for this readout.
- fiducialFrame - The frame index in the fiducial image file where the fiducial frame can be found.
- fiducialColor - The illumination color that is used to measure the fiducials.

## Codebook

The codebook specifies 

## Position list

A csv file containing a list of positions is expected either in the raw data directory or the parent directory. Alternatively, the name of a position csv file that is located in POSITION\_HOME can be provided.

 This text file should contain 'positions' in the file name. The i'th row in the file should be coordinates of the i'th field of view. Each row should contain the x position and the y position with a comma in between.

## Microscope parameters

Microscope parameters specify properties specific to the image acquisition. The microscope parameter file should be place in the MICROSCOPE_PARAMETERS_HOME directory. The parameters that can be set are:

- microns_per_pixel - the number of microns corresponding to one pixel in the image.
- flip_horizontal - flag indicating whether the images should be flipped horizontally in order to align with neighboring images.
- flip_vertical - flag indicating whether the images should be flipped vertically in order to align with neighboring images.
- transpose - flag indicating whether the images should be transposed in order to align with neighboring images.

# Installation

## Specifying paths with a .env file

A .env file is required to specify the search locations for the various input and output files. The following variables should be defined in a file named .env in the project root directory:

- DATA\_HOME - The path of the root directory to the raw data.
- ANALYSIS\_HOME - The path of the root directory where analysis results should be stored.
- CODEBOOK\_HOME - The path of the directory where codebooks can be found.
- DATA\_ORGANIZATION\_HOME - The path of the directory where data organization files can be found.
- POSITION\_HOME - The path of the directory where position files can be found.
- ANALYSIS\_PARAMETERS\_HOME - The path to the directory where analysis parameters can be found.
- MICROSCOPE\_PARAMETERS\_HOME - The path to the directory where microscope parameters can be found.

The contents of an example .env file are below:

```python
DATA_HOME=D:/data
ANALYSIS_HOME=D:/analysis
CODEBOOK_HOME=D:/merfish-parameters/codebooks
DATA_ORGANIZATION_HOME=D:/merfish-parameters/dataorganization
POSITION_HOME=D:/merfish-parameters/positions
ANALYSIS_PARAMETERS_HOME=D:/merfish-parameters/analysis_parameters
MICROSCOPE_PARAMETERS_HOME=D:/merfish-parameters/microscope_parameters
```

## Installing prerequisites

MERlin requires python 3.6 and above. [Storm-analysis](https://github.com/ZhuangLab/storm-analysis) must be intalled prior to installing this package. Additionally, the package rtree is not properly installed by pip and should be installed independently. For example, using Anaconda:

```
conda install rtree
```

## Installing MERlin

MERlin can be install with pip:

```
pip install --process-dependency-links -e MERlin
```

# Usage

## Executing locally

After installation, MERlin can be run from the command line with the input parameters specified, such as: 

```
merlin -d test_data -a test_analysis_parameters -m STORM5 -o Culture_16bits -c HAEC1E1 -n 5
```

Here the images are contained in the directory %DATA\_HOME%\test\_data\ and the analysis tasks listed in test\_analysis\_parameters.json are run with microscope parameters STORM5.json, data organization Culture\_16bits.csv, codebook HAEC1E1 using 5 cores for each process. 

## Executing with Snakemake (Anticipated December 2018)

[Snakemake](https://snakemake.readthedocs.io/en/stable/) is a workflow management system that enables scalable analysis across a wide range of platforms. MERlin can generate a Snakemake workflow that can then be executed using Snakemake. 

## Visualizing the results

The MERlin analysis results can be explored using MERlinView. The analysis of the data set test\_data can be explored with MERlinView from the command line as:

```
merlinview -d test_data
```

## Analysis tasks

### warp.FiducialFitWarp

Description: Aligns image stacks by fitting fiducial spots.

Parameters:
* write\_fiducial\_images - Flag indicating whether the aligned fiducial images
should be saved. These images are helpful for visually verifying the quality 
of the image alignment.

### warp.FiducialCorrelationWarp

Description: Aligns image stacks by maximizing the cross correlation between fiducial images. 

Parameters:
* write\_fiducial\_images - Flag indicating whether the aligned fiducial images
should be saved. These images are helpful for visually verifying the quality 
of the image alignment.

### preprocess.DecovolutionPreprocess

Description: High-pass filters and deconvolves the image data in peparation for bit calling.

Parameters:
* warp\_task - The name of the warp task that provides the aligned image stacks.
* highpass\_pass - The standard deviation to use for the high pass filter.
* decon\_sigma - The standard deviation to use for the lucy richardson deconvolution.

### optimize.Optimize

Description: Determines the optimal per-bit scale factors for barcode decoding.

Parameters:
* iteration\_count - The number of iterations to perform for the optimization.
* fov\_per\_iteration - The number of fields of view to decode in each round of optimization.

### decode.Decode

Description: Extract barcodes from all field of views.

### filterbarcodes.FilterBarcodes

Description: Filters the decoded barcodes based on area and intensity

Parameters:
* area\_threshold - Barcodes with areas below the specified area\_threshold are
removed.
* intensity\_threshold - Barcodes with intensities below this threshold are removed.

### segment.SegmentCells

Description: Determines cell boundaries using a nuclear stain and a cell stain.

Parameters:
* nucleus\_threshold - The relative intensity threshold for seeding the nuclei.
* cell\_threshold - The relative intensity threshold for setting boundaries for the watershed.
* nucleus\_index - The image index for the nucleus image.
* cell\_index - The image index for the cell image.
* z\_index - The z index of the nucleus and cell images to use for segmentation.

### segment.CleanCellSegmentation

Description: Cleans the cell segmentation by merging cells that were fargmented along the field of view boundaries.

### generatemosaic.GenerateMosaic

Description: Assembles the images from each field of view into a low resolution mosaic.

Parameters:
* microns\_per\_pixel - The number of microns to correspond with a pixel in the mosaic.

## Running the tests

## Authors

* **George Emanuel** - *Initial work* 

