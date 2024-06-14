"""
Script used to train ProHMR on egobody RGB dataset and generate smplx model.
Example usage:
python train_prohmr.py --root_dir=/path/to/experiment/folder

Running the above will use the default config file to train ProHMR as in the paper.
The code uses PyTorch Lightning for training.
"""
import os
import argparse
import torch
from tqdm import tqdm
import smplx
from torch.utils.data.dataloader import default_collate
import shutil
import random
from tensorboardX import SummaryWriter

from prohmr.configs import get_config, prohmr_config, dataset_config
from prohmr.models import ProHMRRGBSmplx

from prohmr.datasets.image_dataset_rgb_egobody_smplx import ImageDatasetEgoBodyRgbSmplx, ImageDatasetEgoBodyRgbSmplxMix
from prohmr.datasets.mocap_dataset import MoCapDataset


from utils import *

# python train_prohmr_egobody_rgb_smplx.py --data_source synthetic --train_dataset_root /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/egobody_rgb_new --val_dataset_root /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/egobody_release --train_dataset_file /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/egobody_rgb_new/smplx_spin_holo_rgb_npz/egocapture_train_smplx.npz --val_dataset_file /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/smplx_spin_npz/egocapture_val_smplx.npz
# python train_prohmr_egobody_rgb_smplx.py --load_pretrained True --checkpoint ./data/checkpoint/rgb/best_model.pt --data_source real --train_dataset_root /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/egobody_release --val_dataset_root /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/egobody_release --train_dataset_file /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/smplx_spin_npz/egocapture_train_smplx.npz --val_dataset_file /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/smplx_spin_npz/egocapture_val_smplx.npz

parser = argparse.ArgumentParser(description='ProHMR training code')
parser.add_argument('--gpu_id', type=int, default='0')
parser.add_argument('--load_pretrained', default='Flase', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--load_only_backbone', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='path to save train logs and models')  # data/checkpoint.pt
parser.add_argument('--model_cfg', type=str, default='prohmr/configs/prohmr.yaml', help='Path to config file')  # prohmr prohmr_onlytransl
parser.add_argument('--save_dir', type=str, default='tmp', help='path to save train logs and models')

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=8, help='# of dataloadeer num_workers')
parser.add_argument('--num_epoch', type=int, default=100000, help='# of training epochs ')
parser.add_argument("--log_step", default=100, type=int, help='log after n iters')
parser.add_argument("--save_step", default=100, type=int, help='save models after n iters')

parser.add_argument('--add_bbox_scale', type=float, default=1.2, help='add scale to orig bbox')

parser.add_argument('--with_focal_length', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--with_bbox_info', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--with_cam_center', default='True', type=lambda x: x.lower() in ['true', '1'])

parser.add_argument('--with_full_2d_loss', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--with_global_3d_loss', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--with_transl_loss', default='False', type=lambda x: x.lower() in ['true', '1'])

parser.add_argument('--with_vfov', default='False', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--with_joint_vis', default='False', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--do_augment', default='True', type=lambda x: x.lower() in ['true', '1'])  # todo
parser.add_argument('--shuffle', default='True', type=lambda x: x.lower() in ['true', '1'])  # todo

parser.add_argument('--data_source', type=str, default='mix') 
parser.add_argument('--train_dataset_root', type=str)  
parser.add_argument('--val_dataset_root', type=str)
parser.add_argument('--mix_dataset_root', type=str) 
parser.add_argument('--train_dataset_file', type=str, default=None)  
parser.add_argument('--val_dataset_file', type=str, default=None)
parser.add_argument('--mix_dataset_file', type=str, default=None)  


parser.add_argument('--is_aug', default='True', type=lambda x: x.lower() in ['true', '1'])

args = parser.parse_args()

# args.shuffle = False
torch.cuda.set_device(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('gpu id:', torch.cuda.current_device())

def collate_fn(item):
    try:
        item = default_collate(item)
    except Exception as e:
        import pdb;
        pdb.set_trace()
    return item

def train(writer, logger):
    if args.with_focal_length and args.with_vfov:
        print('[Error] with_focal_length and with_vfov cannot be True at the same time.')
        exit()

    model_cfg = get_config(args.model_cfg)

    # Create dataset and data loader
    if args.data_source != "mix":
        train_dataset = ImageDatasetEgoBodyRgbSmplx(cfg=model_cfg, train=True, device=device, img_dir=args.train_dataset_root,
                                            dataset_file=args.train_dataset_file,
                                            add_scale=args.add_bbox_scale,
                                            do_augment=args.do_augment, data_source=args.data_source, is_train = True, is_aug = args.is_aug)
    else:
        train_dataset = ImageDatasetEgoBodyRgbSmplxMix(cfg=model_cfg, train=True, device=device, 
                                            real_img_dir=args.train_dataset_root,
                                             syn_img_dir=args.mix_dataset_root,
                                            real_dataset_file=args.train_dataset_file,
                                            syn_dataset_file=args.mix_dataset_file,
                                            add_scale=args.add_bbox_scale,
                                            do_augment=args.do_augment, data_source=args.data_source, is_train = True, is_aug = args.is_aug)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, collate_fn=collate_fn)
    train_dataloader_iter = iter(train_dataloader)


    val_dataset = ImageDatasetEgoBodyRgbSmplx(cfg=model_cfg, train=False, device=device, img_dir=args.val_dataset_root,
                                      dataset_file=args.val_dataset_file,            
                                      spacing=1, add_scale=args.add_bbox_scale, data_source="real", is_train = False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    mocap_dataset = MoCapDataset(dataset_file='data/datasets/cmu_mocap.npz')
    mocap_dataloader = torch.utils.data.DataLoader(mocap_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    mocap_dataloader_iter = iter(mocap_dataloader)


    # Setup model
    model = ProHMRRGBSmplx(cfg=model_cfg, device=device, writer=None, logger=None,
                          with_focal_length=args.with_focal_length, with_bbox_info=args.with_bbox_info, with_cam_center=args.with_cam_center,
                          with_full_2d_loss=args.with_full_2d_loss, with_global_3d_loss=args.with_global_3d_loss, with_transl_loss=args.with_transl_loss,
                          with_vfov=args.with_vfov, with_joint_vis=args.with_joint_vis)
    model.train()
    if args.load_pretrained:
        weights = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        if args.load_only_backbone:
            weights_backbone = {}
            weights_backbone['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] == 'backbone'}
            model.load_state_dict(weights_backbone['state_dict'], strict=False)
        else:
            model.load_state_dict(weights['state_dict'], strict=False)
        print('[INFO] pretrained model loaded from {}.'.format(args.checkpoint))
        print('[INFO] load_only_backbone: {}'.format(args.load_only_backbone))
        
    # optimizer
    model.init_optimizers()

    ################################## start training #########################################
    total_steps = 0
    best_loss_keypoints_3d_mode = 10000
    best_loss_keypoints_3d_mode_global = 10000
    for epoch in range(args.num_epoch):
        # for step, batch in tqdm(enumerate(train_dataloader)):
        #     total_steps += 1
        for step in tqdm(range(train_dataset.dataset_len // args.batch_size)):
            total_steps += 1

            ### iter over train loader and mocap data loader
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(train_dataloader)
                batch = next(train_dataloader_iter)

            try:
                mocap_batch = next(mocap_dataloader_iter)
            except StopIteration:
                mocap_dataloader_iter = iter(mocap_dataloader)
                mocap_batch = next(mocap_dataloader_iter)

            # import pdb; pdb.set_trace()
            for param_name in batch.keys():
                if param_name not in ['imgname', 'smpl_params', 'has_smpl_params', 'smpl_params_is_axis_angle']:
                    batch[param_name] = batch[param_name].to(device)
            for param_name in batch['smpl_params'].keys():
                batch['smpl_params'][param_name] = batch['smpl_params'][param_name].to(device)

            for param_name in mocap_batch.keys():
                mocap_batch[param_name] = mocap_batch[param_name].to(device)
            
            output = model.training_step(batch, mocap_batch)
            ####################### log train loss ############################
            if total_steps % args.log_step == 0:
                for key in output['losses'].keys():
                    writer.add_scalar('train/{}'.format(key), output['losses'][key].item(), total_steps)
                    print_str = '[Step {:d}/ Epoch {:d}] [train]  {}: {:.10f}'. \
                        format(step, epoch, key, output['losses'][key].item())
                    logger.info(print_str)
                    print(print_str)

            ####################### log val loss #################################
            if total_steps % args.log_step == 0:
                val_loss_dict = {}
                with torch.no_grad():
                    for test_step, test_batch in tqdm(enumerate(val_dataloader)):
                        for param_name in test_batch.keys():
                            if param_name not in ['imgname', 'smpl_params', 'has_smpl_params', 'smpl_params_is_axis_angle']:
                                test_batch[param_name] = test_batch[param_name].to(device)
                        for param_name in test_batch['smpl_params'].keys():
                            test_batch['smpl_params'][param_name] = test_batch['smpl_params'][param_name].to(device)

                        val_output = model.validation_step(test_batch)

                        for key in val_output['losses'].keys():
                            if test_step == 0:
                                val_loss_dict[key] = val_output['losses'][key].detach().clone()
                            else:
                                val_loss_dict[key] += val_output['losses'][key].detach().clone()

                for key in val_loss_dict.keys():
                    val_loss_dict[key] = val_loss_dict[key] / test_step
                    writer.add_scalar('val/{}'.format(key), val_loss_dict[key].item(), total_steps)
                    print_str = '[Step {:d}/ Epoch {:d}] [test]  {}: {:.10f}'. \
                        format(step, epoch, key, val_loss_dict[key].item())
                    logger.info(print_str)
                    print(print_str)

                # save model with best loss_keypoints_3d_mode
                if val_loss_dict['loss_keypoints_3d_mode'] < best_loss_keypoints_3d_mode:
                    best_loss_keypoints_3d_mode = val_loss_dict['loss_keypoints_3d_mode']
                    save_path = os.path.join(writer.file_writer.get_logdir(), "best_model.pt")
                    state = {
                        "state_dict": model.state_dict(),
                    }
                    torch.save(state, save_path)
                    logger.info('[*] best model saved\n')
                    print('[*] best model saved\n')
                if val_loss_dict['loss_keypoints_3d_full_mode'] < best_loss_keypoints_3d_mode_global:
                    best_loss_keypoints_3d_mode_global = val_loss_dict['loss_keypoints_3d_full_mode']
                    save_path = os.path.join(writer.file_writer.get_logdir(), "best_global_model.pt")
                    state = {
                        "state_dict": model.state_dict(),
                    }
                    torch.save(state, save_path)
                    logger.info('[*] best global model saved\n')
                    print('[*] best global model saved\n')

            ################### save trained model #######################
            if total_steps % args.save_step == 0:
                save_path = os.path.join(writer.file_writer.get_logdir(), "last_model.pt")
                state = {
                    "state_dict": model.state_dict(),
                }
                torch.save(state, save_path)
                logger.info('[*] last model saved\n')
                print('[*] last model saved\n')






if __name__ == '__main__':
    ########## set up writter, logger
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))  # create new path
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))
    sys.stdout.flush()
    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file
    save_config(logdir, args)
    # train_config_file_name = args.model_cfg.split('/')[-1]
    shutil.copyfile(args.model_cfg, os.path.join(logdir, args.model_cfg.split('/')[-1]))

    train(writer, logger)





