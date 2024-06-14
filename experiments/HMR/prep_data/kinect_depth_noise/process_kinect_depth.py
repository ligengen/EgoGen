import cv2
import numpy as np
import matplotlib.pyplot as plt

from kinect_depth_noise.simkinect_camera_utils import add_gaussian_shifts, filterDisp

   
def add_kinect_noise_to_depth(depth_orig, 
                            dot_pattern,
                            scale_factor=100, 
                            baseline_m=0.075, 
                            std=0.5,
                            size_filt=6,
                            focal_length = 554.0,
                            invalid_disp = 99999999.9,
                            a_min = 0.01, #near_plane
                            a_max = 20, #far_plane
                            w = 640,
                            h = 480):

        #depth_orig = np.load(depth_np_path) #(480, 640), min: 1.2863141, max:inf
        depth_invalid_exch = depth_orig.copy()
        depth_invalid_exch[depth_invalid_exch==float('inf')] = invalid_disp
        depth_clipped = depth_invalid_exch
        depth_clipped = np.clip(depth_orig, a_min=a_min, a_max=a_max)
        depth_interp = add_gaussian_shifts(depth_clipped, std=std)
        disp_= focal_length * baseline_m / (depth_interp + 1e-10)
        depth_f = np.round(disp_ * 8.0)/8.0
        out_disp = filterDisp(depth_f, dot_pattern, invalid_disp, size_filt_=size_filt)
        depth = focal_length * baseline_m / (out_disp + 1e-10)
        depth[out_disp == invalid_disp] = 0
        #depth[out_disp == invalid_disp] = float('inf') 
        
        # The depth here needs to be converted to cms so scale factor is introduced 
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
        noisy_depth = (35130/np.round((35130/(np.round(depth*scale_factor) + 1e-10)) + np.random.normal(size=(h, w))*(1.0/6.0) + 0.5))/scale_factor 

        noisy_depth[noisy_depth<a_min] = float('inf')
        noisy_depth[noisy_depth>=a_max] = float('inf')
        return noisy_depth, depth_clipped



if __name__=='__main__':

    w, h = 640, 480
    near_plane = 0.01, #meters
    far_plane = 5, #meters
    focal_length = 554.0

    # simulation params
    scale_factor=100
    baseline_m=0.075
    std=0.5
    size_filt=6


    dot_pattern = cv2.imread("kinect-pattern_3x3.png", 0)

    path_depth_orig = 'example_depth.npy' # 'path/to/orig/depth/image'
    path_noisy_depth_save = 'example_depth_w_noise.npy'

    img_depth = np.load(path_depth_orig)

    #img_depth_w_kinect_noise is the depth with noise, orig_depth_clipped is only for visualization purposes
    img_depth_w_kinect_noise, orig_depth_clipped = add_kinect_noise_to_depth(img_depth, dot_pattern,
                                                                                scale_factor=scale_factor, 
                                                                                baseline_m=baseline_m, 
                                                                                std=std,
                                                                                size_filt=size_filt,
                                                                                focal_length = focal_length,
                                                                                a_min = near_plane,
                                                                                a_max = far_plane,
                                                                                w = w,
                                                                                h = h)
    np.save(path_noisy_depth_save, img_depth_w_kinect_noise)
    
    save_visualization_depth_image_before_after_sim = True
    if save_visualization_depth_image_before_after_sim:
        # the part below is only for visualization, just to replace 'inf' values with the far_plane value.  
        img_depth_w_kinect_noise[img_depth_w_kinect_noise==float('-inf')] = float('inf')
        img_depth_w_kinect_noise[img_depth_w_kinect_noise==float('inf')] = far_plane

        orig_depth_clipped[orig_depth_clipped==float('-inf')] = float('inf')
        orig_depth_clipped[orig_depth_clipped==float('inf')] = far_plane

        plt.imsave('comparison.png', np.hstack((orig_depth_clipped, img_depth_w_kinect_noise)), cmap='magma')

        