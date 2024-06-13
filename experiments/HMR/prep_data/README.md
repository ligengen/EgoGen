## Training Data Preparation

Run the following files to generate the training data. Please remember to replace the data root in each file with your data path.

```
# Generate npz files for Egobody depth images.
python utils_01_gen_egobody_rgb_npz.py

# Add motion blur to EgoGen depth images.
python utils_02_gen_egogen_rgb_add_blur.py

# Generate npz files for EgoGen depth images.
python utils_03_gen_egogen_rgn_npz.py

# Generate npz files for Egobody RGB images.
python utils_04_gen_egobody_depth_npz.py

# Add noise to EgoGen RGB images.
python utils_05_gen_egogen_depth_add_noise.py

# Generate npz files for EgoGen RGB images.
python utils_06_gen_egogen_depth_npz.py
```