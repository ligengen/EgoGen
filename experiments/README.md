## Automated clothing simulation & rendering

### Installation

- [EgoBody scene mesh](https://polybox.ethz.ch/index.php/s/ivcPIYvsCcCIgs4). (For more EgoBody scenes please visit [here](https://sanweiliti.github.io/egobody/egobody.html). You may need to process the scene mesh to Z-axis up and zero floor height by yourself.)
- [HOOD pre-packed environment]().

Organize them as following:
```
EgoGen
  ├── motion/
  ├── experiments/
  |         ├── exp_data/
  |         |      ├── seminar_d78/
```

### Depth image generation

We simulate the EgoBody data collection process as two people switching locations in [EgoBody](https://sanweiliti.github.io/egobody/egobody.html) scenes.

```
cd experiments/
python gen_egobody_depth.py
```
Synthetic data will be saved at `experiments/tmp/`. Change its path in `motion/crowd_ppo/crowd_env_egobody_eval.py`.

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

We sample body textures and 3D clothing meshes from [BEDLAM](https://bedlam.is.tue.mpg.de/). And perform automated clothing simulation leveraging [HOOD](https://dolorousrtur.github.io/hood/).

```

```

## HMR with EgoGen synthetic data

