#! /usr/bin/env python

"""
runs the manual tests without the need to build and install
"""

import os
import sys
import shutil
import logging

sys.path.insert(0, "..")

import wikiboost

def test_setup(config):
  import cctbx.miller
  from iotbx import mtz, pdb
  from scitbx.array_family import flex

  mtz_name = config['mtz_filename']
  mtz_file = mtz.object(mtz_filename)

  pdb_name = config['pdb_name']
  pdb_inp = pdb.input(file_name=pdb_name)
  structure = pdb_inp.xray_structure_simple()
  miller = structure.structure_factors(d_min=2.85).f_calc()
  miller_sub = miller[20000:20002]

  flex.random_generator.seed(82364)
  size = miller.size()
  rand_sel_1 = flex.random_bool(size, 0.5)
  rand_sel_2 = flex.random_bool(size, 0.5)
  miller_1 = miller.select(rand_sel_1).randomize_phases()
  miller_2 = miller.select(rand_sel_2).randomize_phases()
  rand_doub_1 = flex.random_double(miller_1.size(), 0.1) + 0.015
  rand_doub_2 = flex.random_double(miller_2.size(), 0.1) + 0.015
  sigmas_1 = rand_doub_1 * miller_1.amplitudes().data()
  sigmas_2 = rand_doub_2 * miller_2.amplitudes().data()
  miller_1.set_sigmas(sigmas_1)
  miller_2.set_sigmas(sigmas_2)
  miller_1.set_observation_type_xray_amplitude()
  miller_2.set_observation_type_xray_amplitude()
  miller_1.as_intensity_array().i_over_sig_i()
  miller_2.as_intensity_array().i_over_sig_i()

  binner = miller.setup_binner(n_bins=20)
  indices = miller.indices()

  mtch_indcs = miller_1.match_indices(miller_2)
  mset = miller.set()

  module = ["cctbx", "miller"]

  return (mtch_indcs, module)
  

if __name__ == "__main__":

  logging.basicConfig(level=logging.DEBUG)

  logging.info("%%%%%%%%%%%%%%%%%%%% testing %%%%%%%%%%%%%%%%%%%%")
  logging.info("Running _wikiboost()")

  config = {"source_root":
                "/usr/local/cctbx-svn/sources/cctbx_project"
            "mtz_filename": "toxd.mtz",
            "pdb_name": '1imh.pdb',
            "require_params": True}

  boost_object, module = test_setup(config)
  boost_type = type(boost_object)
  doc = wikiboost.wikify_all_methods(boost_type, config, module)

  print doc
