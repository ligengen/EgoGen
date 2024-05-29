import pdb
from pathlib import Path
import os
from omegaconf import DictConfig, OmegaConf, open_dict
import yaml
from models.models_GAMMA_primitive import GAMMAPrimitiveComboGenOP

class ConfigCreator(object):
    def __init__(self, cfg_name):
        self.cfg_name = cfg_name
        exppath = 'crowd_ppo' 
        expname = os.path.basename(exppath)
        # TODO
        # cfg_file = './crowd_ppo/cfg/{:s}.yml'.format(cfg_name)
        cfg_file = './crowd_ppo/cfg_samp20/{:s}.yml'.format(cfg_name)
        try:
            cfg = yaml.safe_load(open(cfg_file, 'r'))
        except FileNotFoundError as e:
            print(e)
            sys.exit()

        # create dirs
        self.cfg_exp_dir = os.path.join('results', expname, cfg_name)
        self.cfg_result_dir = os.path.join(self.cfg_exp_dir, 'results')
        self.cfg_ckpt_dir = os.path.join(self.cfg_exp_dir, 'checkpoints')
        self.cfg_log_dir = os.path.join(self.cfg_exp_dir, 'logs')
        os.makedirs(self.cfg_result_dir, exist_ok=True)
        os.makedirs(self.cfg_ckpt_dir, exist_ok=True)
        os.makedirs(self.cfg_log_dir, exist_ok=True)

        # specify missed experiment settings
        cfg['trainconfig']['save_dir'] = self.cfg_ckpt_dir
        cfg['trainconfig']['log_dir'] = self.cfg_log_dir

        # set subconfigs
        self.modelconfig = cfg['modelconfig']
        self.lossconfig = cfg['lossconfig']
        self.trainconfig = cfg['trainconfig']

def create_dirs(cfg):
    with open_dict(cfg):
        # create dirs
        cfg.cfg_exp_dir = os.path.join('results', 'crowd_ppo', cfg.cfg_name, cfg.wandb.name)
        cfg.cfg_result_dir = os.path.join(cfg.cfg_exp_dir, 'results')
        cfg.cfg_ckpt_dir = os.path.join(cfg.cfg_exp_dir, 'checkpoints')
        cfg.cfg_log_dir = os.path.join(cfg.cfg_exp_dir, 'logs')
        os.makedirs(cfg.cfg_result_dir, exist_ok=True)
        os.makedirs(cfg.cfg_ckpt_dir, exist_ok=True)
        os.makedirs(cfg.cfg_log_dir, exist_ok=True)

        # specify missed experiment settings
        cfg['trainconfig']['save_dir'] = cfg.cfg_ckpt_dir
        cfg['trainconfig']['log_dir'] = cfg.cfg_log_dir

def configure_model(cfg, gpu_index, seed):
    cfgall = ConfigCreator(cfg)
    modelcfg = cfgall.modelconfig
    traincfg = cfgall.trainconfig
    predictorcfg = ConfigCreator(modelcfg['predictor_config'])
    regressorcfg = ConfigCreator(modelcfg['regressor_config'])

    testcfg = {}
    testcfg['gpu_index'] = gpu_index
    testcfg['ckpt_dir'] = traincfg['save_dir']
    testcfg['result_dir'] = cfgall.cfg_result_dir
    testcfg['seed'] = seed
    testcfg['log_dir'] = cfgall.cfg_log_dir
    testop = GAMMAPrimitiveComboGenOP(predictorcfg, regressorcfg, testcfg)
    testop.build_model(load_pretrained_model=True)

    return testop

def load_model():
    # cfg = OmegaConf.load('crowd_ppo/cfg/MPVAEPolicy_samp_collision.yaml')
    # TODO
    cfg = OmegaConf.load('crowd_ppo/cfg_samp20/MPVAEPolicy_samp_collision.yaml')
    create_dirs(cfg)
    OmegaConf.save(cfg, Path(cfg.cfg_exp_dir, "config.yaml"))
    cfg_policy = cfg
    args = cfg.args
    # cfg_1frame_male = cfg_policy.trainconfig['cfg_1frame_male']
    cfg_2frame_male = cfg_policy.trainconfig['cfg_2frame_male']
    # cfg_1frame_female = cfg_policy.trainconfig['cfg_1frame_female']
    cfg_2frame_female = cfg_policy.trainconfig['cfg_2frame_female']

    """set GAMMA primitive networks"""
    # genop_1frame_male = configure_model(cfg_1frame_male, args.gpu_index, args.random_seed)
    # genop_1frame_female = configure_model(cfg_1frame_female, args.gpu_index, args.random_seed)
    genop_2frame_male = configure_model(cfg_2frame_male, args.gpu_index, args.random_seed)
    genop_2frame_female = configure_model(cfg_2frame_female, args.gpu_index, args.random_seed)

    return cfg, genop_2frame_male, genop_2frame_female


if __name__ == '__main__':
    x,y,z,w=load_model()
    pdb.set_trace()
