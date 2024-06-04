#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
import pdb
from pathlib import Path
import pickle
import json

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.env import DummyVectorEnv

import sys
sys.setrecursionlimit(10000) # if too small, deepcopy will reach the maximal depth limit.
sys.path.append(os.getcwd())

from crowd_ppo.crowd_env_2f_box import CrowdEnv
from crowd_ppo.primitive_model import load_model
from crowd_ppo.ppo_policy import GAMMAPPOPolicy
from exp_GAMMAPrimitive.utils.environments import BatchGeneratorScene2frameTrainBox as SceneTrainGen
from exp_GAMMAPrimitive.utils import config_env
from models.baseops import SMPLXParser
from models.models_policy_ppo import GAMMAActor, GAMMACritic, GAMMAPolicyBase
from models.models_policy_ppo import ActorCritic
from human_body_prior.tools.model_loader import load_vposer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="collision-avoidance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=20000) # # transitions collected per epoch.
    parser.add_argument("--step-per-collect", type=int, default=1024) # # transitions the collector would collect before the network update 
    parser.add_argument("--repeat-per-collect", type=int, default=1) # max_train_iter_1/2f
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=256)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--rew-norm", type=int, default=False) # same as gamma
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument("--vf-coef", type=float, default=1.0) # same as gamma
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--weight-kld", type=float, default=0) # gamma: 10
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="clip")
    # parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.1) # same as cfg
    parser.add_argument("--eps-clip", type=float, default=0.1)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=1) # same as gamma
    parser.add_argument("--recompute-adv", type=int, default=0) # 
    parser.add_argument("--logdir", type=str, default="./log/log_box")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-buffer", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument(
        "--dynobs",
        default=False,
        action="store_true",
        help="evaluation on dynamic obstacle",
    )
    parser.add_argument('--more-ego', default=False, action='store_true', help="more egosensing dim")
    parser.add_argument(
        "--deterministic-eval",
        default=False,
        action="store_true",
    )
    return parser.parse_args()


def main(args=get_args()):

    env = CrowdEnv(init_env)
    train_envs = DummyVectorEnv([lambda: CrowdEnv(init_env) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: CrowdEnv(init_env) for _ in range(args.test_num)])

    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model separate gru and map encoder here
    actor = GAMMAActor(init_env[0].modelconfig).to(args.device)
    critic = GAMMACritic(init_env[0].modelconfig).to(args.device)
    shared_net = GAMMAPolicyBase(init_env[0].modelconfig).to(args.device)

    actor_critic = ActorCritic(actor, critic, shared_net)

    # init params
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
        # GRU param initialization?
        # if isinstance(m, torch.nn.GRU):
        #     torch.nn.init.zeros_(m.bias_ih_l0)
        #     torch.nn.init.zeros_(m.bias_hh_l0)
        #     m.weight_ih_l0.data.copy_(0.01 * m.weight_ih_l0.data)
        #     m.weight_hh_l0.data.copy_(0.01 * m.weight_hh_l0.data)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor_critic.actor.pnet.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.AdamW(actor_critic.parameters(), lr=args.lr, weight_decay=0.01)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = GAMMAPPOPolicy(
        actor,
        critic,
        shared_net,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        weight_kld=args.weight_kld,
        reward_normalization=args.rew_norm,
        # do not clip/modify action
        action_space=None,
        action_scaling=False,
        action_bound_method="",
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        deterministic_eval=args.deterministic_eval
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt["model"])
        # optim.load_state_dict(ckpt["optim"])
        # train_envs.set_obs_rms(ckpt["obs_rms"])
        # test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", args.resume_path)
        if args.resume_buffer:
            train_collector.buffer = pickle.load(open(args.resume_buffer, "rb"))
            print("Loaded buffer from: ", args.resume_buffer)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)

    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ppo"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer, save_interval=args.save_interval)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        state = {"model": policy.state_dict()} # , "optim": optim.state_dict()}#, "obs_rms": train_envs.get_obs_rms()}
        torch.save(state, os.path.join(log_path, "policy.pth"))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        # torch.save({"model": policy.state_dict(), "optim": optim.state_dict()}, ckpt_path)
        torch.save({"model": policy.state_dict()}, ckpt_path)
        # buffer_path = os.path.join(log_path, f"train_buffer_{epoch}.pkl")
        # pickle.dump(train_collector.buffer, open(buffer_path, "wb"))
        return ckpt_path


    if not args.watch:
        # trainer
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            step_per_collect=args.step_per_collect,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False,
            save_checkpoint_fn=save_checkpoint_fn,
        )
        pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    # init env 
    cfg, \
    genop_2frame_male, \
    genop_2frame_female = load_model(box=True)
    # dataset
    scene_list_path = Path('data/scenes/random_box_obstacle_new_names.pkl')
    with open(scene_list_path, 'rb') as f:
        scene_list = pickle.load(f)
    print('#scene: ', len(scene_list))
    bm_path = config_env.get_body_model_path()

    vposer, _ = load_vposer(bm_path + '/vposer_v1_0', vp_model='snapshot')
    vposer.eval()
    vposer.to('cuda')

    # TODO: should rewrite batch_gen_amass.BatchGeneratorReachingTarget not load smplx and vposer everytime
    motion_seed_dir = "data/locomotion"
    motion_seed_list = list(Path(motion_seed_dir).glob('*npz'))
    scene_sampler = SceneTrainGen(dataset_path='', motion_seed_list=motion_seed_list,
                                   scene_dir='data/scenes', scene_list=scene_list,
                                   scene_type='random_box_obstacle_new', body_model_path=bm_path)

    # smplx parser
    pconfig_1frame = {
        'n_batch': 1 * 4,
        'device': 'cuda',
        'marker_placement': 'ssm2_67'
    }
    smplxparser_1frame = SMPLXParser(pconfig_1frame)

    pconfig_2frame = {
        'n_batch': 2 * 4,
        'device': 'cuda',
        'marker_placement': 'ssm2_67'
    }
    smplxparser_2frame = SMPLXParser(pconfig_2frame)

    pconfig_mp = {
        'n_batch': 20 * 4,
        'device': 'cuda',
        'marker_placement': 'ssm2_67'
    }
    smplxparser_mp = SMPLXParser(pconfig_mp)


    with open(config_env.get_body_marker_path() + '/SSM2.json') as f:
        marker_ssm_67 = json.load(f)['markersets'][0]['indices']
    feet_markers = ['RHEE', 'RTOE', 'RRSTBEEF', 'LHEE', 'LTOE', 'LRSTBEEF']
    feet_marker_idx = [list(marker_ssm_67.keys()).index(marker_name) for marker_name in feet_markers]
    body_markers = list(marker_ssm_67.values())

    init_env = [cfg, genop_2frame_male, genop_2frame_female, \
                bm_path, scene_sampler, smplxparser_1frame, smplxparser_2frame, \
                smplxparser_mp, feet_marker_idx, body_markers, vposer]
    main()
