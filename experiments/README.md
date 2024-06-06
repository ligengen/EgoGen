## Automated clothing simulation & rendering

### Installation

- Download [EgoBody scene mesh](https://polybox.ethz.ch/index.php/s/ivcPIYvsCcCIgs4). (For more EgoBody scenes please visit [here](https://sanweiliti.github.io/egobody/egobody.html). You may need to process the scene mesh to Z-axis up and zero floor height by yourself.)
- Sign in [BEDLAM](https://bedlam.is.tue.mpg.de/index.html). Download [clothing textures (diffuse and normal map)](https://bedlam.is.tue.mpg.de/clothingsim.php).
- Download [HOOD data](https://polybox.ethz.ch/index.php/s/6kbnyuG4HG9wyWg).

Organize them as following:
```
EgoGen
  ├── motion/
  ├── experiments/
  |         ├── exp_data/
  |         |      └── seminar_d78/
  |         |       
  |         ├── HOOD/
  |               └── hood_data
  |                       ├── bedlam/
  |                              ├── clothing_textures/ # from BEDLAM
  |                              ├── ...
```
Make sure you install HOOD conda environment:
```
cd HOOD
conda env create -f hood.yml
```

### Depth image generation

We simulate the EgoBody data collection process as two people switching locations in [EgoBody](https://sanweiliti.github.io/egobody/egobody.html) scenes.

```
# Make sure you are still using egogen environment!

cd experiments/
python gen_egobody_depth.py
```
Synthetic data will be saved at `experiments/tmp/`. Note: We only used zero-shape male motion data to train our motion model, but during this evaluation, we randomly sample body shape, gender, and initial motion seed to increase data diversity. As a result, synthesized motions might have decreased motion quality.

Saved smplx parameters format:
```
0:3 transl
3:6 global_orient
6:69 body_pose
69:85 camera pose inverse matrix (used to transform to egocentric camera frame coordinates)
85:95 beta
95 gender (male = 0, female = 1)
```

### RGB image generation

We sample body textures and 3D clothing meshes from [BEDLAM](https://bedlam.is.tue.mpg.de/). And perform automated clothing simulation leveraging [HOOD](https://dolorousrtur.github.io/hood/). When we develop EgoGen, HOOD only supports T-pose garment meshes as input. So we modified their code by ourselves. You may check their up-to-date repo for their updated implementation. In addition, we also modified their supported body model from smpl to smplx.

```
# First make sure you are still using egogen environment!
# And you need to replace HOOD_PYTHON with your hood env python path

python gen_egobody_rgb.py
```
Synthetic data will be saved at `experiments/tmp/`. Saved smplx parameters format:
```
0:96 same as depth
96 cx (camera intrinsics)
97 cy
98 fx(fy)
```

Regarding how to use our synthetic data to train an HMR model, please refer to `utils_gen_depth_npz.py` and `utils_gen_rgb_npz.py`

### Add more clothing meshes

Due to limitations of HOOD, we could only do separate simulation of top garments and pants. Besides, it only support clothing meshes that can be represented as a connected graph. In our implementation, the initial pose of garments is A-pose (same as BEDLAM).

You may add more clothing meshes by refering to [our script](HOOD/new_clothes.py) and their repo. You may need to separate a single garment mesh from BEDLAM to upper and lower meshes by yourself.

## HMR with EgoGen synthetic data

