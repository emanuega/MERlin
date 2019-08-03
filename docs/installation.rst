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

Storm-analysis_ must be installed prior to installing this package. Additionally, the packages rtree and pytables are not properly installed by pip and should be installed independently. For example, using Anaconda:

.. _Storm-analysis: https://github.com/ZhuangLab/storm-analysis

.. code-block:: none

    conda install rtree pytables

On Harvard research computing, matplotlib raises an error saying that 'CXXABI_1.3.9' is not found. This can be corrected by:

.. code-block:: none

    module load gcc/8.2.0-fasrc01
    
Installing MERlin
==================

MERlin can be installed by cloning the repository and installing with pip:

.. code-block:: none

    git clone https://github.com/emanuega/MERlin

.. code-block:: none

    pip install -e MERlin

Specifying paths with a .env file
==================================

A .env file is required to specify the search locations for the various input and output files. The following variables should be defined in a file named .env in the MERlin root directory:

* DATA\_HOME - The path of the root directory to the raw data.
* ANALYSIS\_HOME - The path of the root directory where analysis results should be stored.
* PARAMETERS\_HOME - The path to the directory where the merfish-parameters directory resides.

The contents of an example .env file are below:

.. code-block:: none

    DATA_HOME=D:/data
    ANALYSIS_HOME=D:/analysis
    PARAMETERS_HOME=D:/merfish-parameters
