import numpy as np


from merlin.core import analysistask


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







class GaussianFitting(analysistask.ParallelAnalysisTask):

    """
    An abstract class for obtaining the positions and intensities of fluorescence
    spots using a 3D gaussian function as model. 
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'write_fiducial_images' not in self.parameters:
            self.parameters['write_fiducial_images'] = False
        if 'write_aligned_images' not in self.parameters:
            self.parameters['write_aligned_images'] = False

        self.writeAlignedFiducialImages = self.parameters[
                'write_fiducial_images']