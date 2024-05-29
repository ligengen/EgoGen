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
- [] Motion model eval code for dynamic obstacle avoidance and crowd motion synthesis
- [] Motion primitive C-VAE training code
- [] Motion model training code
- [] Egocentric human mesh recovery code (RGB/depth images as input)
- [] EgoBody synthetic data (RGB/depth)
- [] EgoBody synthetic data generation script (incl. automated clothing simulation)
- [] EgoGen rendering pipeline code


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

Organize as following:
```
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
|   └── ...
├── results/    # C-VAE pretrained models
```

## Inference

### Ego-perception driven motion synthesis in crowded scenes
```
python -W ignore crowd_ppo/main_ppo.py --resume-path=data/checkpoint_87.pth --watch --deterministic-eval
```
This will generate a virtual human walking in the replica room0 with sampled `(start, target)` location. Generated motion sequences are located in `log/eval_results/`

### Motion visualization

```
python vis.py --path motion_seq_path
```

## Stay Tuned ...

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
