import os
from utils.validation import Config as ValidationConfig
from utils.validation import load_runner_from_checkpoint, update_config_for_validation, create_one_sequence_dataloader
from utils.arguments import load_params
from utils.common import move2device, pickle_dump
from utils.defaults import DEFAULTS
from pathlib import Path
import argparse
import pdb
os.environ["HOOD_PROJECT"] = "../HOOD"
os.environ["HOOD_DATA"] = "./hood_data"
HOOD_PROJECT = os.environ["HOOD_PROJECT"]
HOOD_DATA = os.environ["HOOD_DATA"]

def hood_eval(sequence_path, garment_name, batch_size):
    # Set material paramenters, see configs/cvpr.yaml for the training ranges for each parameter
    config_dict = dict()
    config_dict['density'] = 0.20022
    config_dict['lame_mu'] = 23600.0
    config_dict['lame_lambda'] = 44400
    config_dict['bending_coeff'] = 3.962e-05
    config_dict['separate_arms'] = False
    config_dict['batch_size'] = int(batch_size)

    config_dict['garment_dict_file'] = './hood_data/bedlam/garments_dict.pkl'
    if args.gender == "female":
        config_dict['smpl_model'] = '../data/smplx/models/smplx/SMPLX_FEMALE.npz'
    elif args.gender == "male":
        config_dict['smpl_model'] = '../data/smplx/models/smplx/SMPLX_MALE.npz'

    validation_config = ValidationConfig(**config_dict)


    # Choose the model and the configuration file

    config_name = 'postcvpr'
    checkpoint_path = Path('hood_data') / 'trained_models' / 'postcvpr.pth'


    # garments_dict_path = os.path.join(DEFAULTS.aux_data, 'garments_dict.pkl')

    # load the config from .yaml file and load .py modules specified there
    modules, experiment_config = load_params(config_name)

    # modify the config to use it in validation 
    experiment_config = update_config_for_validation(experiment_config, validation_config)

    # load Runner object and the .py module it is declared in
    runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)

    # name of the garment to sumulate
    # garment_name = 'bedlam_top'

    dataloader = create_one_sequence_dataloader(os.path.join(HOOD_DATA, sequence_path), 
                                                garment_name, modules, experiment_config)
    sequence = next(iter(dataloader))
    sequence = move2device(sequence, 'cuda:0')

    trajectories_dict = runner.valid_rollout(sequence,  bare=True)

    # Save the sequence to disc
    out_path = os.path.join(DEFAULTS.data_root, 'temp', 'output_%s.pkl' % garment_name)
    pickle_dump(dict(trajectories_dict), out_path)
    
    """
    from utils.show import write_video 
    from aitviewer.headless import HeadlessRenderer

    # Careful!: creating more that one renderer in a single session causes an error
    renderer = HeadlessRenderer()
    out_video = Path(DEFAULTS.data_root) / 'temp' / 'output.mp4'
    write_video(out_path, out_video, renderer)
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--garment-name', help='Description for foo argument', required=True)
    parser.add_argument('--seq-name', help='Description for bar argument', required=True)
    parser.add_argument('--bz', help='Description for bar argument', required=True)
    parser.add_argument('--gender', help='', required=True)
    args = parser.parse_args()
    hood_eval(args.seq_name, args.garment_name, args.bz)
