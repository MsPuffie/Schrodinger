===========
Schrodinger
===========

.. image:: https://travis-ci.org/MsPuffie/Schrodinger.svg?branch=master
    :target: https://travis-ci.org/MsPuffie/Schrodinger
    
.. image:: https://img.shields.io/pypi/v/schrodinger.svg
        :target: https://pypi.python.org/pypi/schrodinger

.. image:: https://img.shields.io/travis/MsPuffie/schrodinger.svg
        :target: https://travis-ci.org/MsPuffie/schrodinger

.. image:: https://readthedocs.org/projects/schrodinger/badge/?version=latest
        :target: https://schrodinger.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/MsPuffie/schrodinger/shield.svg
     :target: https://pyup.io/repos/github/MsPuffie/schrodinger/
     :alt: Updates



This Project Solves the 1-step Schrodinger equation numeriacally, using tensorflow.
    ˆ HΨ(x) = EΨ(x)
The deﬁnition of ˆ HΨ(x) is
    −c∇2Ψ(x) + V0(x) 
where V0 is the potential energy, c is a constant, and ∇2 is the Laplacian.



* Free software: MIT license
* Documentation: https://schrodinger.readthedocs.io.


Installation
--------

- pip install tensorflow
- pip install coverage


* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
