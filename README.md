# <p align="center">EgoGen: An Egocentric Synthetic Data Generator </p>

####  <p align="center"> [Gen Li](https://vlg.inf.ethz.ch/team/Gen-Li.html), [Kaifeng Zhao](https://vlg.inf.ethz.ch/team/Kaifeng-Zhao.html), [Siwei Zhang](https://vlg.inf.ethz.ch/team/Siwei-Zhang.html), [Xiaozhong Lyu](https://vlg.inf.ethz.ch/team/Xiaozhong-Lyu.html), [Mihai Dusmanu](https://dusmanu.com/), [Yan Zhang](https://yz-cnsdqz.github.io/), [Marc Pollefeys](https://people.inf.ethz.ch/marc.pollefeys/), [Siyu Tang](https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html)</p>

<p align="center">
    <img src="assets/logos.svg" alt="Logo" height="40">
</p>

### <p align="center">[ArXiv](https://arxiv.org/abs/2401.08739) | [Project Page](https://ego-gen.github.io/)

### <p align="center"> CVPR 2024 (Oral)

<p align="center">
  <img width="100%" src="assets/teaser.jpg"/>
</p><p align="center">
  <b>EgoGen</b>: a scalable synthetic data generation system for egocentric perception tasks, with rich multi-modal data and accurate annotations. We simulate camera rigs for head-mounted devices (HMDs) and render from the perspective of the camera wearer with various sensors. Top to bottom: middle and right camera sensors in the rig. Left to right: photo-realistic RGB image, RGB with simulated motion blur, depth map, surface normal, segmentation mask, and world position for fisheye cameras widely used in HMDs.
</p>



## Release Plan

We will release all code before the CVPR 2024 conference.

- [x] Motion model eval code in Replica room0
- [x] Motion model training code (two-stage RL in crowded scenes)
- [x] Motion model eval code for crowd motion synthesis
- [x] Motion model training code (models for dynamic evaluations)
- [x] Motion primitive C-VAE training code
- [ ] Egocentric human mesh recovery code (RGB/depth images as input)
- [ ] EgoBody synthetic data (RGB/depth)
- [x] EgoBody synthetic data generation script (incl. automated clothing simulation)
- [x] EgoGen rendering pipeline code


## Installation

### Environment

Download the packed conda environment [here](https://polybox.ethz.ch/index.php/s/rciHeHNVuLmlWif). **Note**: As modifications have been made to some packages in the environment, we do not provide `req.txt` or `env.yml`.

Please install `conda-pack` to unpack the environment:
```
mkdir -p egogen
tar -xzf egogen.tar.gz -C egogen
source egogen/bin/activate
```

The code is tested on Ubuntu 22.04, CUDA 11.7.

### Extra Models and Data

- [SMPL-X body model and VPoser](https://smpl-x.is.tue.mpg.de/)
- [Precomputed sdf for replica room0](https://polybox.ethz.ch/index.php/s/qFxEWnMHXtzMB5N)
- [Pretrained marker regressor and predictor model (C-VAE)](https://polybox.ethz.ch/index.php/s/Ss4YwjR5s6EfuX6)
- [Pretrained policy model for motion synthesis in Replica room0](https://polybox.ethz.ch/index.php/s/aA9A5D8DLVli2a2)
- [Pretrained policy model for motion synthesis in dynamic settings](https://polybox.ethz.ch/index.php/s/us7JcDPqxSVaT7F)
- [Static box scenes for policy training](https://polybox.ethz.ch/index.php/s/n8PIyHZVmXFl9Sr)
- [SAMP Mocap dataset](https://samp.is.tue.mpg.de/)

Organize them as following:
```
EgoGen
  ├── motion
        ├── crowd_ppo/
        ├── data/
        |   ├── smplx/
        |   │   └── models/
        |   |       |── smplx/
        |   |       |   |── SMPLX_MALE.npz
        |   |       |   |── ...
        |   |       |
        |   |       |── vposer_v1_0/
        |   |       |   |── snapshots/TR00_E096.pt
        |   |       |   |── ...
        |   ├── room0_sdf.pkl
        |   ├── checkpoint_87.pth
        |   ├── checkpoint_best.pth
        |   ├── scenes/
        |   |      ├── random_box_obstacle_new/
        |   |      └── random_box_obstacle_new_names.pkl
        |   ├── samp/*_stageII.pkl  # original samp dataset
        |   └── ...
        ├── results/    # C-VAE pretrained models
```

## Data preparation

SAMP dataset is processed to motion primitive format with these two commands:
```
python exp_GAMMAPrimitive/utils/utils_canonicalize_samp.py 1
python exp_GAMMAPrimitive/utils/utils_canonicalize_samp.py 10
cp -r data/samp/Canonicalized-MP/data/locomotion data/
```
Processed files will be located at `data/samp/Canonicalized-MP*/`. And copy locomotion data for initial motion seed sampling in policy training.

## Inference

### Ego-perception driven motion synthesis in crowded scenes
```
cd motion
python -W ignore crowd_ppo/main_ppo.py --resume-path=data/checkpoint_87.pth --watch --deterministic-eval
```
This will generate a virtual human walking in the replica room0 with sampled `(start, target)` location. Generated motion sequences are located in `log/eval_results/`

####  Motion visualization

```
python vis.py --path motion_seq_path
```

### Ego-perception driven motion synthesis for crowd motion
The pretrained model is trained in scenes with a single static box obstacle. And it is directly generalizable to dynamic settings. We release the code for four humans switching locations. You can easily modify the code for two/eight humans crowd motion synthesis. 
```
cd motion
python crowd_ppo/main_crowd_eval.py (--deterministic-eval)
```

`--deterministic-eval` is optional. If you want to synthesize more diverse motions, do not add it. You may also randomly sample initial motion seed to further increase diversity. Generated motion sequences are located in `log/eval_results/crowd-4human/*`

#### Motion visualization
```
python vis_crowd.py --path 'log/eval_results/crowd-4human/*'
```

## Training

### Ego-perception driven motion synthesis in crowded scenes

#### Phase 1: RL pretraining with soft penetration termination
```
cd motion
python -W ignore crowd_ppo/main_ppo.py
```
We selected `checkpoint_113.pth` as the best pretrained model. 

Principles to choose the model: (1) the reward is high and the kld loss is small; (2) choose models with smaller epoch numbers if their rewards are similar. These principles will make sure the learned action space does not deviate too much from the prior, and as a result producing more natural motions.

#### Phase 2: RL finetuning with strict penetration termination
```
python -W ignore crowd_ppo/main_ppo.py --resume-path=/path/to/pretrained/checkpoint_113.pth --logdir=log/finetune/ --finetune
```
This would produce `log/finetune/checkpoint_87.pth` that you downloaded before. The best model should have (1) high reward; (2) small kld loss.

### Ego-perception driven motion synthesis for crowd motion

Our egocentric perception driven motion primitives exhibit remarkable generalizability. The model is training with static scenes but can be used to synthesize crowd motions. To train such model:

```
cd motion
python crowd_ppo/main_ppo_box.py
```
This would produce models with similar performance as `checkpoint_best.pth` (its test reward was 10.22), which should be the trained `log/log_box/checkpoint_164.pth`. Using this model, you can synthesize human motions in dynamic settings.

### Motion primitive C-VAE

Our action space is the latent space (128-D Gaussian) of this C-VAE. 

Make sure you did "Data preparation" section.
The body marker predictor (history markers + action -> future markers) can be trained as:
```
python exp_GAMMAPrimitive/train_GAMMAPredictor.py --cfg MPVAE_samp20_2frame

python exp_GAMMAPrimitive/train_GAMMAPredictor.py --cfg MPVAE_samp20_2frame_rollout --resume_training 1

# The above command will raise FileExistsError. Copy the last ckpt from MPVAE_samp20_2frame to MPVAE_samp20_2frame_rollout as epoch 0:
cp results/exp_GAMMAPrimitive/MPVAE_samp20_2frame/checkpoints/epoch-300.ckp results/exp_GAMMAPrimitive/MPVAE_samp20_2frame_rollout/checkpoints/epoch-000.ckp

# And run it again:
python exp_GAMMAPrimitive/train_GAMMAPredictor.py --cfg MPVAE_samp20_2frame_rollout --resume_training 1
```
The final trained model `results/exp_GAMMAPrimitive/MPVAE_samp20_2frame_rollout/checkpoints/epoch-400.ckp` is the pretrained `results/crowd_ppo/MPVAE_samp20_2frame_rollout/checkpoints/epoch-400.ckp`.

For body marker regressor (markers -> body mesh), we use the pretrained model from [GAMMA](https://github.com/yz-cnsdqz/GAMMA-release).

## Automated clothing simulation & rendering

Refer to [EgoBody synthetic data generation script](./experiments/README.md). We use Pyrender in this script for faster rendering speed, which may reduce photorealism. If you require higher quality please check EgoGen rendering module.

## EgoGen rendering module

Please download the blender file [here](https://polybox.ethz.ch/index.php/s/DHESUq5zm6DxNN0). Some notes of the code:

* When importing `.pkl` motion sequences to blender using the script, please click "Scene Collection" first, then run the script.
* `render.py`: convert `.npz` files generated by blender to images.
* `vid.sh`: convert images to videos.
* Blender dependencies: [smpl-x blender addon](https://smpl-x.is.tue.mpg.de/) and [vision blender](https://github.com/Cartucho/vision_blender).

Tested on Blender 3.4.1 Linux x64.

## License

* Third-party software and datasets employs their respective license. Here are some examples:
  * SMPL-X body model follows its own license.
  * AMASS dataset follows its own license.
  * Blender and its SMPL-X add-on employ their respective license.

* The rests employ the **Apache 2.0 license**.

## Citation
```
@inproceedings{li2024egogen, 
 title={{EgoGen: An Egocentric Synthetic Data Generator}}, 
 author={Li, Gen and Zhao, Kaifeng and Zhang, Siwei and Lyu, Xiaozhong and Dusmanu, Mihai and Zhang, Yan and Pollefeys, Marc and Tang, Siyu}, 
 booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
 year={2024} 
}
```
