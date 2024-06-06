import os
import socket

from munch import munchify

hostname = socket.gethostname()

HOOD_PROJECT = '../HOOD' 
HOOD_DATA = './hood_data'

DEFAULTS = dict()

DEFAULTS['server'] = 'local'
DEFAULTS['data_root'] = HOOD_DATA
DEFAULTS['experiment_root'] = os.path.join(HOOD_DATA, 'experiments')
DEFAULTS['vto_root'] = os.path.join(HOOD_DATA, 'vto_dataset')
DEFAULTS['aux_data'] = ""
DEFAULTS['project_dir'] = HOOD_PROJECT

# TODO: change ts_scale to 1
DEFAULTS['hostname'] = hostname
DEFAULTS = munchify(DEFAULTS)
