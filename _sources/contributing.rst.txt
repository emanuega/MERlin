Contributing to MERlin
************************

Contributions to MERlin can either be submitted by opening an issue to raise concerns or offer suggestions or by opening a pull request to offer improvements to the code base.  

Opening a pull request
========================

A pull request allows code to be proposed to be incorporated into MERlin. To receive feedback on work in progress, mark the pull request with WIP in the subject line. To open a pull request:

#. Fork the repository to your github account and clone it locally.
#. Create a new branch for your edits.
#. Make your desired edits to the code.
#. Run the tests to ensure MERlin is still functional. Write new tests to cover your new contribution as necessary. 
#. Submit a pull request from your edited branch to the master branch of the MERlin repository. Be sure to reference any relevant issues. 


Code formatting
===============

Code contributions should follow the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>` style guide with the
exception that variable names should be mixedCase instead of words separated by underscores. Comments should follow
the `Google docstring style <http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`.

Running the tests
=================

All contributions to MERlin must maintain the integrity of the tests. Before submitting a pull request, please ensure
that all tests pass. Tests are implemented using the pytest_ framework. The tests are in the test directory and they
can be run by executing pytest in the root MERlin directory. To facilitate efficient debugging, tests that take more
than few seconds are marked with ```slowtest``` and can be excluded from the run using the command:

.. _pytest: https://docs.pytest.org/

.. code-block:: none

    pytest -v  test

Generating documentation
=============================

Documentation for MERlin is generated using Sphinx. The API documentation can be generated with the command from the root MERlin directory:

.. code-block:: none

    sphinx-apidoc -f -o ./docs/_modules .
