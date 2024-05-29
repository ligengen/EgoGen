import yaml
import os, sys
import socket

def get_host_name():
    return hostname


def get_body_model_path():
    if 'vlg-atlas' in hostname:
        bmpath = '/vlg-nfs/kaizhao/datasets/models_smplx_v1_1/models/'
    elif 'dalcowks' in hostname:
        bmpath = '/home/kaizhao/dataset/models_smplx_v1_1/models/'
    else:
        # raise ValueError('not stored here')
        pass
    bmpath = 'data/smplx/models'
    return bmpath

def get_body_marker_path():
    if 'vlg-atlas' in hostname:
        mkpath = '/vlg-nfs/kaizhao/datasets/models_smplx_v1_1/models/markers'
    elif 'dalcowks' in hostname:
        mkpath = '/home/kaizhao/dataset/models_smplx_v1_1/models/markers'
    else:
        pass
        # raise ValueError('not stored here')
    mkpath = 'data'
    return mkpath

def get_amass_canonicalized_path():
    if 'vlg-atlas' in hostname:
        mkpath = '/vlg-nfs/kaizhao/datasets/amass/AMASS-Canonicalized-MP/data'
    elif 'dalcowks' in hostname:
        mkpath = '/home/kaizhao/dataset/amass/AMASS-Canonicalized-locomotion-MP/data'
    else:
        # raise ValueError('not stored here')
        pass
        mkpath = '/home/genli/Desktop/GAMMA-release/AMASS-Canonicalized-locomotion-MP/data'
    # mkpath = '/local/home/genligen/gamma_interaction/gamma_release/AMASS-Canonicalized-locomotion-MP/data'
    return mkpath

def get_amass_canonicalizedx10_path():
    if 'vlg-atlas' in hostname:
        mkpath = '/vlg-nfs/kaizhao/datasets/amass/AMASS-Canonicalized-MPx10/data'
    elif 'dalcowks' in hostname:
        mkpath = '/home/kaizhao/dataset/amass/AMASS-Canonicalized-locomotion-MPx10/data'
    else:
        # raise ValueError('not stored here')
        pass
        mkpath = '/home/genli/Desktop/GAMMA-release/AMASS-Canonicalized-locomotion-MPx10/data'
    # mkpath = '/local/home/genligen/gamma_interaction/gamma_release/AMASS-Canonicalized-locomotion-MPx10/data'
    return mkpath




hostname = socket.gethostname()
print('host name:', hostname)



















