import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util, skinner
import time
import pyximport; pyximport.install()
from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import math
import os
import shutil
import random
import sys
import configparser
import io
import yaml
import pymoab_utils_2 as utpy
from prolongation import ProlongationTPFA3D as prol3D
from restriction import Restriction as rest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from others_utils import OtherUtils as oth
# import upscaling

parent_dir = os.path.dirname(os.path.abspath(__file__))
# parent_parent_dir = os.path.dirname(parent_dir)
# input_dir = os.path.join(parent_parent_dir, 'input')
# flying_dir = os.path.join(parent_parent_dir, 'flying')
# utils_dir = os.path.join(parent_parent_dir, 'utils')
# output_dir = os.path.join(parent_parent_dir, 'output')

name_inputfile = '27x27x27_out.h5m'
principal = '/elliptic'
dir_output = '/elliptic/output'
parent_dir = os.path.dirname(__file__)
out_dir = os.path.join(parent_dir, 'output')


#
# mesh_config_file = 'mesh_configs.cfg'
# config = configparser.ConfigParser()
# config.read(mesh_config_file)
# total_dimension = config['total-dimension']
# Lx = long(total_dimension['Lx'])
# Ly = long(total_dimension['Ly'])
# Lz = long(total_dimension['Lz'])
# import pdb; pdb.set_trace()
