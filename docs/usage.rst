Usage
******

MERlin organizes files into three folders, specified in the .merlinenv file (see :ref:`specifying-paths`). The three folders are:

DATA_HOME – base directory for raw data
ANALYSIS_HOME – base directory for the analysis results
PARAMETERS_HOME – base directory for parameters

MERlin reads raw data, such as images, from DATA_HOME and configuration parameters from PARAMETERS_HOME and writes
analysis results into ANALYSIS_HOME. Each separate experiment should be a separated folder within DATAHOME and MERlin
will create a corresponding folder in ANALYSIS_HOME. For example, the images for “experiment1” should be stored in the
folder %DATA_HOME%/experiment1. When merlin runs, it will save the output files in  %ANALYSIS_HOME%/experiment1. With
this file system, %DATA_HOME% and %ANALYSIS_HOME% are constant for all the experiments you analyze and only the
experiment name needs to be specified. The typical file structure for MERFISH experiments
"experiment1" and "experiment2" could be:

.. code-block:: none

    %DATA_HOME%/
    ├── experiment1/
    │   ├── image_000_00.tif
    │   ├── image_000_01.tif
    │   ├── ...
    │   └── image_150_10.tif
    └── experiment2/
        ├── image_000_00.tif
        ├── image_000_01.tif
        ├── ...
        └── image_150_10.tif
    %PARAMETERS_HOME%/
    ├── analysis/
    │   └── analysis_parameters.json
    ├── codebooks/
    │   └── codebook.csv
    ├── dataorganization/
    │   └── dataorganization.csv
    ├── microscope/
    │   └── microscope_parameters.json
    ├── positions/
    │   └── positions.csv
    └── snakemake/
        └── snakeconfig.json
    %ANALYSIS_HOME%/
    ├── experiment1/
    │   ├── FiducialCorrelationWarp
    │   ├── DeconvolutionPreprocess
    │   ├── ...
    │   └── PlotPerformance
    └── experiment2/
        ├── FiducialCorrelationWarp
        ├── DeconvolutionPreprocess
        ├── ...
        └── PlotPerformance


Input specifications
=====================

Raw images
-----------

All raw images should be indicated in the same folder, as discussed above, and there should be a separated image
file for each imaging round and each field of view. MERlin is able to read both tiff stacks and dax image files. The
exact file name is specified by a regular expression in the data organization file (imageRegExp and fiducialRegExp).
For example, you can specify the regular expression (?P<imageType>[\w|-]+)_(?P<fov>[0-9]+)_(?P<imagingRound>[0-9]+)
for the image filenames specified below. This indicates that the first part of the file is the imageType (the value in
the imageType column of the data organization file), followed by the fov index, followed by imagingRound index, all
separated by an underscore.

Data organization
------------------

The data organization file specifies which the images correspond to each readout. The data organization file is a csv file. The first row is a header with column names and each following row designates one readout. The information provided for each readout indicates where to find the corresponding images in the raw image data and how to find the corresponding fiducial image to align the images between rounds.

The columns in the data organization file are:

- channelName - The name of the data channel. For genes measured sequential, this can be set as the gene name.
- readoutName - The name of the readout sequence used to measure this channel.
- imageType - The base name for the image file that contains the images for this readout, for example, ``Conventional_750_650_561_488_405``
- imageRegExp - A regular expression specifying how image file names are constructed for each field of view and
each imaging round. The parameters used in the regular expression are ``imageType``, ``fov``, and ``imagingRound``,
for example: ``(?P<imageType>[\w|-]+)_(?P<fov>[0-9]+)_(?P<imagingRound>[0-9]+)``. Here, ``imageType`` specifies the
string indicated in the ``imageType`` column for the corresponding row, ``imagingRound`` specifies the designated
``imagingRound`` for the corresponding row, and ``fov`` is filled with all field of view indexes in the data set. The
imageRegExp should not include the file extension, which will be determined automatically.
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

The position list is a csv file containing a list of positions for each imaged region. The i'th row in the file should be
coordinates of the i'th field of view. Each position is specified by the x position and the y position, separated by a comma.

The name of a position csv file that within the POSITION\_HOME directory can be provided, otherwise the positions are
extracted from the image metadata xml.

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

    merlin -a test_decode_and_segment.json -m microscope.json -o 7z_16bits.csv -c L26E1_codebook.csv -n 5 testdata

Here the MERFISH images contained in the directory `%DATA\_HOME%/testdata/` are processed using the analysis tasks listed in `test\_analysis\_parameters.json` with microscope parameters `STORM5.json`, data organization `Culture\_16bits.csv`, codebook `L26E1_codebook.csv` using 5 cores for each process.

Executing on a high performance cluster
=====================================================

MERlin executes tasks through Snakemake_, a workflow management system. Each task can be distributed over a high performance
cluster that is run by a scheduler, such as SLURM or Sge, by indicating the appropriate command to submit the job to snakemake.
See the `merlin-parameters-example <https://github.com/emanuega/merlin-parameters-example>`_ repository for an example snakemake
configuration file. Additional arguments can be specified as indicated in the
`snakemake api documentation <https://snakemake.readthedocs.io/en/stable/api_reference/snakemake.html>`_.

.. code-block:: none

    merlin -a test_decode_and_segment.json -m microscope.json -o 7z_16bits.csv -c L26E1_codebook.csv -k snake.json testdata

.. _Snakemake: https://snakemake.readthedocs.io/en/stable/

