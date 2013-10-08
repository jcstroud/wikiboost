===========
 wikiboost 
===========

Introduction
------------

A python package that parses boost python and converts it to wikitext


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
object named ``mtz_file`` (of type ``iotbx_mtz_ext.object``)
is created along with many other types of objects. Because the ``__name__``
of the class of ``mtz_file`` describes where the class can be
found in the cctbx sources (``iotbx/mtx``), the path to
the module must not be explicitly given. Here, the module
is ``None``. In other cases, the path can be given explicitly
as a list (e.g. ``["cctbx", "miller"]``).  The module (or ``None``)
is returned as the second element of the return value of ``test-setup()``.

The first element of the return value of ``test-setup()``
is the ``mtch_indcs`` object.

The ``wikifiy_all_methods`` function takes three arguments:

1. ``boost_type``: the class or type of the object returned from ``test-setup()``
2. ``config``: a configuration dictionary with the essential keys
       - ``source_root``: absolute path to the cctbx sources
       - ``require_params``: set to ``False`` only in special cases of debugging
   The ``config`` dict will hopefully (soon) be moved to a configuration file that
   is specified using phyles (https://pypi.python.org/pypi/phyles/)
3. ``module``: the elements of the module directory, relative to
   the ``source_root`` as specified in ``config``
