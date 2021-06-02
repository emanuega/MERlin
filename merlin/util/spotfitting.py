# import from other packages
import numpy as np 
import os
import time
from scipy.stats import scoreatpercentile
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.filters import gaussian_filter


# import from this package
from . import _seed_th
from .. import _sigma_zxy
from ..External import Fitting_v4
from ..visual_tools import get_seed_points_base



# import matplotlib.pyplot as plt


"""
This Module contains utility functions for 3d gaussian fitting used for 
chromatin imaging. 

The functions are adapted from Pu Zheng's analysis package found in 
https://github.com/zhengpuas47/ImageAnalysis3. The core fitting function
was written by Bogdan Bintu. 


"""

# spots = ia.spot_tools.fitting.fit_fov_image(I,channel_num, max_num_seeds=2000)

"""
fit_fov_image



	get_seeds

		remove_edge_points

	Fitting_v4

		iter_fit_seed_points

		first_fit

			closest_faster

			Gaussian_fit

				to_natural_paramaters

		repeat_fit

			Gaussian_fit

		find_image_background

		 generate_neighboring_crop
"""




# integrated function to get seeds
def get_seeds(imageStackIn:  np.ndarray,      #im, 
			  maxNumSeeds: int,  #max_num_seeds=None, 
			  seedIntensityThreshold=150.0, #th_seed=150, 
              usePercentile=False,# use_percentile=False,
              seedPercentileThreshold=95.0, #th_seed_per=95, 
              selectedCenter=None, # sel_center=None, 
              seedRadius=30, #seed_radius=30,
              maximumFilterSigma=0.75, #gfilt_size=0.75, 
              minimumFilterSigma=8.0, # 8.0 
              filterSize=3, #3, 
              minimumEdgeDistance=3, #=2,#min_edge_distance=2,
              useDynamicThreshold=True,#use_dynamic_th=True, 
              numDynamicIters=10, #=10, #dynamic_niters=10,
              minDynamicSeeds=1, =1,# min_dynamic_seeds=1,
              removeHotPixels=True,#remove_hot_pixel=True, 
              hotPixelThreshold=3, #=3, #hot_pixel_th=3,
              returnSeedHeight=False, #return_h=False, 
              verbose=False) -> = np.ndarray:
              
    """
    paramter names to be given to merlin

	seed_max_num
	seet_intensity_threshold
	seed_percentile_threshold
	seed_use_percentile
	seed_selected_center
	

	"""

    """
    Function to fully get seeding pixels given a image and thresholds.
	
	Args:
		imageStaskIn: a 3-dimensional numpy array arranged as (z,x,y)
		maxNumSeeds: (int) maximum number of seeds to generate.
    	seedIntensityThreshold: (float) seeding threshold between max_filter -
    	 	min_filter 
      	usePercentile: (bool) whether to use percentile to determine 
      		seedIntensityThreshold
      	seedPercentileThreshold: (float) seeding percentile for intensities, 	
      sel_center: selected center coordinate to get seeds, array-like, same dimension as im, 
        default=None (whole image), 
      seed_radius: square frame radius of getting seeds, int, default=30,
      gfilt_size: gaussian filter size for max_filter image, float, default=0.75, 
      background_gfilt_size: gaussian filter size for min_filter image, float, default=10,
      filt_size: filter size for max/min filter, int, default=3, 
      min_edge_distance: minimal allowed distance for seed to image edges, int/float, default=3,
      use_dynamic_th: whetaher use dynamic th_seed, bool, default=True, 
      dynamic_niters: number of iterations used for dynamic th_seed, int, default=10, 
      min_dynamic_seeds: minimal number of seeds to get with dynamic seeding, int, default=1,
      return_h: whether return height of seeds, bool, default=False, 
      verbose: whether say something!, bool, default=False,
    """
    # check inputs

  
    if th_seed_per >= 100 or th_seed_per <= 50:
        use_percentile = False
        print(f"th_seed_per should be a percentile > 50, invalid value given ({th_seed_per}), so not use percentile here.")
    # # crop image if sel_center is given
    # if sel_center is not None:
    #     if len(sel_center) != len(np.shape(im)):
    #         raise IndexError(f"num of dimensions should match for selected center and image given.")
    #     # get selected center and cropping neighbors
    #     _center = np.array(sel_center, dtype=np.int)
    #     _llims = np.max([np.zeros(len(im.shape)), _center-seed_radius], axis=0)
    #     _rlims = np.min([np.array(im.shape), _center+seed_radius], axis=0)
    #     _lims = np.array(np.transpose(np.stack([_llims, _rlims])), dtype=np.int)
    #     _lim_crops = tuple([slice(_l,_r) for _l,_r in _lims])
    #     # crop image
    #     _im = im[_lim_crops].copy()
    #     # get local centers for adjustment
    #     _local_edges = _llims
    # else:
    #     _local_edges = np.zeros(len(np.shape(im)))
    #     _im = im.copy()
        

    # get threshold
    if usePercentile:
        _th_seed = scoreatpercentile(imageStackIn, seedPercentileThreshold) -
         		   scoreatpercentile(imageStackIn, 
         		   						(100-seedPercentileThreshold)/2)
    else:
        _th_seed = th_seed
    if verbose:
        _start_time = time.time()
        if not use_dynamic_th:
            print(f"-- start seeding image with threshold: {_th_seed:.2f}", end='; ')
        else:
            print(f"-- start seeding image, th={_th_seed:.2f}", end='')
    

    ## do seeding
    if not use_dynamic_th:
        dynamic_niters = 1 # setting only do seeding once
    else:
        dynamic_niters = int(dynamic_niters)
    # front filter:
    if gfilt_size:
        _max_im = np.array(gaussian_filter(_im, gfilt_size), dtype=np.float)
    else:
        _max_im = _im.astype(np.float)
    _max_ft = np.array(maximum_filter(_max_im, int(filt_size)), dtype=np.float)
    # background filter
    if background_gfilt_size:
        _min_im = np.array(gaussian_filter(_im, background_gfilt_size), dtype=np.float)
    else:
        _min_im = _im.astype(np.float)
    _min_ft = np.array(minimum_filter(_min_im, int(filt_size)), dtype=np.float)

    #return _max_ft, _min_ft, _max_im, _min_im

    # iteratively select seeds
    for _iter in range(dynamic_niters):
        # get seed coords
        _current_seed_th = _th_seed * (1-_iter/dynamic_niters)
        #print(_iter, _current_seed_th)
        # should be: local max, not local min, differences large than threshold
        _coords = np.where((_max_ft == _max_im) & (_min_ft != _min_im) & (_max_ft-_min_ft >= _current_seed_th))
        # remove edges
        if min_edge_distance > 0:
            _keep_flags = remove_edge_points(_im, _coords, min_edge_distance)
            _coords = tuple(_cs[_keep_flags] for _cs in _coords)
        # if got enough seeds, proceed.
        if len(_coords[0]) >= min_dynamic_seeds:
            break

    # print current th
    if verbose and use_dynamic_th:
        print(f"->{_current_seed_th:.2f}", end=', ')
    # hot pixels
    if remove_hot_pixel:
        _,_x,_y = _coords
        _xy_str = [str([np.round(x_,1),np.round(y_,1)]) 
                    for x_,y_ in zip(_x,_y)]
        _unique_xy_str, _cts = np.unique(_xy_str, return_counts=True)
        _keep_hot = np.array([_xy not in _unique_xy_str[_cts>=hot_pixel_th] 
                             for _xy in _xy_str],dtype=bool)
        _coords = tuple(_cs[_keep_hot] for _cs in _coords)
    # get heights
    _hs = (_max_ft - _min_ft)[_coords]
    _final_coords = np.array(_coords) + _local_edges[:, np.newaxis] # adjust to absolute coordinates
    if return_h: # patch heights if returning it
        _final_coords = np.concatenate([_final_coords, _hs[np.newaxis,:]])
    # transpose and sort by intensity decreasing order
    _final_coords = np.transpose(_final_coords)[np.flipud(np.argsort(_hs))]
    if verbose:
        print(f"found {len(_final_coords)} seeds in {time.time()-_start_time:.2f}s")
    # truncate with max_num_seeds
    if max_num_seeds is not None and max_num_seeds > 0 and max_num_seeds <= len(_final_coords):
        _final_coords = _final_coords[:np.int(max_num_seeds)]
        if verbose:
            print(f"--- {max_num_seeds} seeds are kept.")
    
    return _final_coords



