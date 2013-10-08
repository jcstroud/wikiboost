===========
 wikiboost 
===========

Introduction
------------

A python that parses boost python and converts it to wikitext


Home Page & Repository
----------------------

Home Page (eventually): http://wikiboost.bravais.net

Repository: https://github.com/jcstroud/wikiboost


Example
-------

A usage example is in the ``test`` directory in the
file ``wikiboost-test.py``.

Usage consists roughly of

1. Construct an object of the class that needs to be documented
2. Call ``wikiboost.wikify_all_methods()`` on the ``type()`` of that object

In ``wikiboost-test.py``, step (1) above is accomplished via
the function ``test-setup()``. As of version 0.1.0, the
object named ``mtch_indcs`` (of type ``ctbx_miller_ext.match_indices``)
is created along with many other types of objects. Because the ``__name__``
of the class of ``mtch_indcs`` has an "ext" in it, the path to
the module must be explicitly given. Here, the module
is ``["cctbx", "miller"]``. This module is returned as the second
element of the return value of ``test-setup()``. The first
element is the ``mtch_indcs`` object.

The ``wikifiy_all_methods`` function takes three arguments:

1. ``boost_type``: the class or type of the object returned from ``test-setup()``
2. ``config``: a configuration dictionary with the essential keys
       - ``source_root``: absolute path to the cctbx sources
       - ``require_params``: set to ``False`` only in special cases of debugging
   The ``config`` dict will hopefully (soon) be moved to a configuration file that
   is specified using phyles (https://pypi.python.org/pypi/phyles/)
3. ``module``: the elements of the module directory, relative to
   the ``source_root`` as specified in ``config``
