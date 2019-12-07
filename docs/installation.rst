Installation
**************
    
Set up a virtual environment
=============================

To ensure that Merlin and its dependencies don't interfere with other packages that are installed, we recommend that you install MERlin in a new virtual environment. MERlin requires python 3.6 or above. 

An anaconda virtual environment can be created using the command:

.. code-block:: none

    conda create -n merlin_env python=3.6

and the new environment can be activated using the command:

.. code-block:: none

    conda activate merlin_env

or 

.. code-block:: none

    source activate merlin_env

Installing prerequisites
==========================

The packages rtree and pytables are not properly installed by pip and should be installed independently. For example, using Anaconda:

.. code-block:: none

    conda install rtree pytables

On Harvard research computing, matplotlib raises an error saying that 'CXXABI_1.3.9' is not found. This can be corrected by loading the gcc module:

.. code-block:: none

    module load gcc/8.2.0-fasrc01
    
Installing MERlin
==================

MERlin can be installed by cloning the repository and installing with pip:

.. code-block:: none

    git clone https://github.com/emanuega/MERlin

.. code-block:: none

    pip install -e MERlin


.. _specifying-paths:

Specifying paths with a .env file
==================================

A .merlinenv file is required to specify the search locations for the various input and output files. The following variables should be defined in a file named .merlinenv in the user home directory (~\\.merlinenv on linux or C:\\users\\UserName\\.merlinenv on Windows):

* DATA\_HOME - The path of the root directory to the raw data.
* ANALYSIS\_HOME - The path of the root directory where analysis results should be stored.
* PARAMETERS\_HOME - The path to the directory where the merfish-parameters directory resides.

The PARAMETERS_HOME directory should contain the following folders:

* analysis - Contains the analysis parameters json files.
* codebooks - Contains the codebook csv files.
* dataorganization - Contains the data organization csv files.
* positions - Contains the position csv files.
* microscope - Contains the microscope parameters json files.
* fpkm - Contains the fpkm csv files.
* snakemake - Contains the snakemake arguments json files.

An example PARAMETERS_HOME directory with typical files can be found in the
`merlin-parameters-example <https://github.com/emanuega/merlin-parameters-example>`_ repository.

The contents of an example .merlinenv file are below:

.. code-block:: none

    DATA_HOME=D:/data
    ANALYSIS_HOME=D:/analysis
    PARAMETERS_HOME=D:/merfish-parameters

Merlin can create a .merlinenv file for you using the command:

.. code-blocks:: none

    merlin --configure .
