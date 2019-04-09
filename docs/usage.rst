Usage
******

Input specifications
=====================

Raw images
-----------

Data organization
------------------

The data organization file specifies which the images correspond to each readout. The data organization file is a csv file. The first row is a header with column names and each following row designates one readout. The information provided for each readout indicates where to find the corresponding images in the raw image data and how to find the corresponding fiducial image to align the images between rounds.

The columns in the data organization file are:

- bitName - The name of the readout.
- imageType - The base name for the image file that contains the images for this readout, for example, ``Conventional_750_650_561_488_405``
- imageRegExp - A regular expression specifying how image file names are constructed for each field of view and each imaging round. The parameters used in the regular expression are ``imageType``, ``fov``, and ``imagingRound``, for example: ``(?P<imageType>[\w|-]+)_(?P<fov>[0-9]+)_(?P<imagingRound>[0-9]+)``. Here, ``imageType`` specifies the string indicated in the ``imageType`` column for the corresponding row, ``imagingRound`` specifies the designated ``imagingRound`` for the corresponding row, and ``fov`` is filled with all field of view indexes in the data set. 
- bitNumber - The bit number corresponding to this readout.
- imagingRound - The round of imaging where this readout is measured, starting from zero.
- color - The illumination color that is used to measure this readout.
- frame - The zero indexed frame or frames in the image file where images corresponding to this readout can be found. For a single frame, a single integer can be provided. For multiple frames, the frames can be provided as a list as ``[0, 1, 2, 3, 4, 5, 6]``
- zPos - The z position for each of the frames specified in the previous column. For only a single frame, the z position should be provided as a decimal number while for multiple frames a list should be provided as for frame.
- fiducialImageType - The base name for the image file that contains the fiducial images for aligning images this readout, for example, ``Conventional_750_650_561_488_405``
- fiducialRegExp - A regular expression specifying how file names are constructed for the fiducial image files. This regex follows the same format as ``imageRegExp``.
- fiducialImagingRound - The imaging round (zero indexed) corresponding to the fiducial images for aligning images for this readout.
- fiducialFrame - The frame index in the fiducial image file where the fiducial frame can be found.
- fiducialColor - The illumination color that is used to measure the fiducials.

Codebook
----------

The codebook specifies . Barcodes corresponding to blanks must have "blank" in their name.

Position list
--------------

The position list is a csv file containing a list of positions for each imaged region. The i'th row in the file should be coordinates of the i'th field of view. Each position is specified by the x position and the y position, separated by a comma. 

This file is expected to contain 'positions' in the file name and should be present either in the raw data directory or the parent directory. Alternatively, the name of a position csv file that is located in POSITION\_HOME can be provided. 

Microscope parameters
-----------------------

Microscope parameters specify properties specific to the image acquisition. The microscope parameter file should be placed in the MICROSCOPE_PARAMETERS_HOME directory. The parameters that can be set are:

- microns_per_pixel - the number of microns corresponding to one pixel in the image.
- flip_horizontal - flag indicating whether the images should be flipped horizontally in order to align with neighboring images.
- flip_vertical - flag indicating whether the images should be flipped vertically in order to align with neighboring images.
- transpose - flag indicating whether the images should be transposed in order to align with neighboring images.


Executing locally
===================

After installation, MERlin can be run from the command line with the input parameters specified, such as: 

.. code-block:: none

    merlin -a test_analysis_parameters.json -m STORM5.json -o Culture_16bits.csv -c HAEC1E1.csv -n 5 testdata

Here the MERFISH images contained in the directory `%DATA\_HOME%/testdata/` are processed using the analysis tasks listed in `test\_analysis\_parameters.json` with microscope parameters `STORM5.json`, data organization `Culture\_16bits.csv`, codebook `HAEC1E1` using 5 cores for each process. 

Executing with Snakemake (Anticipated December 2018)
=====================================================

Snakemake_ is a workflow management system that enables scalable analysis across a wide range of platforms. MERlin can generate a Snakemake workflow that can then be executed using Snakemake. 

.. _Snakemake: https://snakemake.readthedocs.io/en/stable/

Visualizing the results
========================

The MERlin analysis results can be explored using MERlinView. The analysis of the data set test\_data can be explored with MERlinView from the command line as:

.. code-block:: none

    merlinview -d test_data


