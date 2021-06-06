# import from other packages
import numpy as np 
import os
import time
from scipy.stats import scoreatpercentile
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.filters import gaussian_filter


# import from this package
# from . import _seed_th
# from .. import _sigma_zxy
# from ..External import Fitting_v4
# from ..visual_tools import get_seed_points_base



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

# def remove_edge_points(im, T_seeds, distance=2):
    
#     im_size = np.array(np.shape(im))
#     _seeds = np.array(T_seeds)[:len(im_size),:].transpose()
#     flags = []
#     for _seed in _seeds:
#         _f = ((_seed >= distance) * (_seed <= im_size-distance)).all()
#         flags.append(_f)
    
#     return np.array(flags, dtype=np.bool)


# integrated function to get seeds
def get_peaks(imageStackIn:  np.ndarray,      #im, 
              maxNumPeaks: int,  #max_num_seeds=None, 
              peakIntensityThreshold=150.0, #th_seed=150, 
              usePercentile=False,# use_percentile=False,
              peakPercentileThreshold=95.0, #th_seed_per=95,  
              maxFilterSigma=0.75, #gfilt_size=0.75, 
              minFilterSigma=8.0, # 8.0 
              filterSize=3, #3, 
              minEdgeDistance=3, #=2,#min_edge_distance=2,
              useDynamicThreshold=True,#use_dynamic_th=True, 
              numDynamicIters=10, #=10, #dynamic_niters=10,
              minDynamicPeaks=1,# min_dynamic_seeds=1,
              removeHotPixels=True,#remove_hot_pixel=True, 
              hotPixelThreshold=3, #=3, #hot_pixel_th=3,
              returnPeakHeights=False, #return_h=False, 
              verbose=False) ->  np.ndarray:
              
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
        maxNumPeaks: (int) maximum number of Peaks to generate. Default=None
        seedIntensityThreshold: (float) seeding threshold between max_filter -
            min_filter 
        usePercentile: (bool) whether to use percentile to determine 
            seedIntensityThreshold
        seedPercentileThreshold: (float) seeding percentile for intensities, 
        maxFilterSigma: (float) gaussian filter sigma for max_filter image
        minFilterSigma: (float) gaussian filter size for min_filter image
        filterSize: (int) filter size for max/min filter
        minEdgeDistance: (float) minimal allowed distance for peak to image
            edges
        useDynamicThreshold: (bool) whether use dynamic peakIntensityThreshold
        numdDynamicIters: (int) number of iterations used for dynamic 
            seedIntensityThreshold 
        minDynamicSeeds: (int) minimal number of seeds to get with dynamic 
            seeding
        returnSeedHeight: (bool) whether return height of seeds,
        verbose: (bool) whether say something!

    Output:
        Nx3 np.array with peak coordinates, in decreasing height order
        if returnSeedHeight = True, returns Nx4 array where last column
        is the peak height.

    """
    if verbose:
        startTime = time.time()


    # check inputs
    if peakPercentileThreshold >= 100 or peakPercentileThreshold <= 50:
        usePercentile = False
        if verbose:
            print(f' peakPercentileThreshold should satisfy: 50 >= '
                  f'peakPercentileThreshold <= 100. Current value ('
                  f'{peakPercentileThreshold}) its outside the'
                  f'accepted range. Not using percentile method.')   
    
    # enforcing integer types
    maxNumPeaks = int(maxNumPeaks)
    filterSize = int(filterSize) 
    numDynamicIters = int(numDynamicIters)
    minDynamicPeaks = int(minDynamicPeaks)
              
    # Get threshold
    if usePercentile:
        threshold = scoreatpercentile(imageStackIn, 
                                      peakPercentileThreshold) - \
                    scoreatpercentile(imageStackIn, 
                                     (100-peakPercentileThreshold)/2)
    else:
        threshold = peakIntensityThreshold


    # Generate maximum filter image
    maxGaussFilter = np.array(gaussian_filter(imageStackIn, 
                                              maxFilterSigma), dtype=np.float)
    imageMaxFilter = np.array(maximum_filter(maxGaussFilter, 
                                             filterSize), dtype=np.float)

    # generate minimum filter
    minGaussFilter = np.array(gaussian_filter(imageStackIn, 
                                              minFilterSigma), dtype=np.float)
    imageMinFilter = np.array(minimum_filter(minGaussFilter, 
                                             filterSize), dtype=np.float)

    """ 
    Select peaks lower higher in intensity than the specified threshold.
    if not enough peaks are founds, lower the threshold iteratively and repeat
    the procedure numDynamicIters times until maxNumPeaks peaks are found.
    """ 
    for i in range(numDynamicIters):

        currentPeakThreshold = threshold * (1-i/numDynamicIters)

        """
        Select pixels that are local maxima, not local minima, and 
        where the difference between the max and filter is larger than the 
        threhold.
        """ 
        peakCoords = np.where( (imageMaxFilter == maxGaussFilter) 
                                & (imageMinFilter != minGaussFilter) 
                                & (imageMaxFilter - imageMinFilter >=
                                        currentPeakThreshold))
        
        """
        remove peaks that are less than minEdgeDistance pixels away from the border of the image.
        """
     
        if minEdgeDistance > 0:
            imShape = np.array(imageStackIn.shape)
            peakCoordsT = np.array(peakCoords).T
            keepIdx = []
            for coords in peakCoordsT:
                idx = ( (coords >= minEdgeDistance) 
                      * (coords <= imShape-minEdgeDistance)).all()
                keepIdx.append(idx)
            peakCoords = tuple(x[keepIdx] for x in peakCoords)
    
   
        # if got enough seeds, proceed.
        if len(peakCoords[0]) >= minDynamicPeaks:
            if verbose:
                print(f' currentPeakThreshold = {currentPeakThreshold}. '
                      f'{len(peakCoords[0])} peaks found so far, exiting.')
            break
        elif verbose:
            print(f' currentPeakThreshold = {currentPeakThreshold}. '
                  f'{len(peakCoords[0])} peaks found so far, decreasing '
                  f'threshold.')

    # Remove hot pixels (repeated peaks)
    if removeHotPixels:
        _,_x,_y = peakCoords
        _xy_str = [str([np.round(x_,1),np.round(y_,1)]) 
                    for x_,y_ in zip(_x,_y)]
        _unique_xy_str, _cts = np.unique(_xy_str, return_counts=True)
        _keep_hot = np.array([_xy not in 
                            _unique_xy_str[_cts>=hotPixelThreshold] 
                             for _xy in _xy_str],dtype=bool)
        peakCoords = tuple(_cs[_keep_hot] for _cs in peakCoords)
    
    # tuple to array, into to float
    localEdges = np.zeros(len(np.shape(imageStackIn)))
    finalPeakCoords = np.array(peakCoords) + _local_edges[:, np.newaxis] 
    
    # get peak heights
    peakHeights = (imageMaxFilter - imageMinFilter)[peakCoords]
    if returnPeakHeights: # patch heights if returning it     
        finalPeakCoords = np.concatenate([finalPeakCoords, 
                                          peakHeights[np.newaxis,:]])
    
    # transpose and sort by intensity decreasing order
    finalpeakCoords = np.transpose(finalpeakCoords)
    finalpeakCoords = finalpeakCoords[np.flipud(np.argsort(peakHeights))]
    
    if verbose:
        print(f' found {len(_finalpeakCoords)} peaks in'
              f' {time.time()-startTime:.2f}s')

    # truncate with maxNumPeaks
    if maxNumPeaks is not None and maxNumPeaks > 0 \
            and maxNumPeaks <= len(_finalpeakCoords):
        finalpeakCoords = _finalpeakCoords[:np.int(maxNumPeaks)]
        if verbose:
            print(f" {maxNumPeaks} peaks are kept.")
    
    return finalPeakCoords




