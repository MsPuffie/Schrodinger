===========
Schrodinger
===========

.. image:: https://travis-ci.org/MsPuffie/Schrodinger.svg?branch=master
    :target: https://travis-ci.org/MsPuffie/Schrodinger
    
.. image:: https://img.shields.io/pypi/v/schrodinger.svg
        :target: https://pypi.python.org/pypi/schrodinger

.. image:: https://readthedocs.org/projects/schrodinger/badge/?version=latest
        :target: https://schrodinger.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



This Project Solves the 1-step Schrodinger equation numeriacally, using tensorflow.
    ˆ HΨ(x) = EΨ(x)
The deﬁnition of ˆ HΨ(x) is
    −c∇2Ψ(x) + V0(x) 
where V0 is the potential energy, c is a constant, and ∇2 is the Laplacian.


Installation
--------

- pip install tensorflow
following this link if you have questions downloading tensorflow:
https://www.tensorflow.org/install/pip?lang=python3
- pip install coverage


Usage
--------
Make sure you aready download tensorflow under python3, then use the following commands to run this project

git clone https://github.com/MsPuffie/Schrodinger
cd schrodinger
python3 schrodinger/schrodinger.py


* Free software: MIT license
* Documentation: https://schrodinger.readthedocs.io.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
