import os
from typing import List
from typing import Dict
from typing import Union
import numpy as np
from skimage import transform
from skimage import feature

import storm_analysis.sa_library.parameters as saparameters
import storm_analysis.daostorm_3d.mufit_analysis as mfit
import storm_analysis.sa_library.sa_h5py as saH5Py

from merlin.core import analysistask
from merlin.util import registration
from merlin.util import aberration


class Warp(analysistask.ParallelAnalysisTask):

    """
    An abstract class for warping a set of images so that the corresponding
    pixels align between images taken in different imaging rounds.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'write_fiducial_images' not in self.parameters:
            self.parameters['write_fiducial_images'] = False

        self.writeAlignedFiducialImages = self.parameters[
                'write_fiducial_images']

    def get_aligned_image_set(
            self, fov: int,
            chromaticCorrector: aberration.ChromaticCorrector=None
    ) -> np.ndarray:
        """Get the set of transformed images for the specified fov.

        Args:
            fov: index of the field of view
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 4-dimensional numpy array containing the aligned images. The
                images are arranged as [channel, zIndex, x, y]
        """
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        zPositions = self.dataSet.get_z_positions()
        return np.array([[self.get_aligned_image(fov, d, z, chromaticCorrector)
                          for z in zPositions] for d in dataChannels])

    def get_aligned_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector=None
    ) -> np.ndarray:
        """Get the specified transformed image

        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
            zIndex: index of the z position
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 2-dimensional numpy array containing the specified image
        """
        inputImage = self.dataSet.get_raw_image(dataChannel, fov, zIndex)
        transformation = self.get_transformation(fov, dataChannel)
        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization()\
                            .get_data_channel_color(dataChannel)
            return transform.warp(chromaticCorrector.transform_image(
                inputImage, imageColor), transformation, preserve_range=True
                ).astype(inputImage.dtype)
        else:
            return transform.warp(inputImage, transformation,
                                  preserve_range=True).astype(inputImage.dtype)

    def _process_transformations(self, transformationList, fov) -> None:
        """
        Process the transformations determined for a given fov. 

        The list of transformation is used to write registered images and 
        the transformation list is archived.

        Args:
            transformationList: A list of transformations that contains a
                transformation for each data channel. 
            fov: The fov that is being transformed.
        """

        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        '''
        zPositions = self.dataSet.get_z_positions()

        imageDescription = self.dataSet.analysis_tiff_description(
                len(zPositions), len(dataChannels))

        with self.dataSet.writer_for_analysis_images(
                self, 'aligned_images', fov) as outputTif:
            for t, x in zip(transformationList, dataChannels):
                for z in zPositions:
                    inputImage = self.dataSet.get_raw_image(x, fov, z)
                    transformedImage = transform.warp(
                            inputImage, t, preserve_range=True) \
                        .astype(inputImage.dtype)
                    outputTif.save(
                            transformedImage, 
                            photometric='MINISBLACK',
                            metadata=imageDescription)
        '''

        if self.writeAlignedFiducialImages:

            fiducialImageDescription = self.dataSet.analysis_tiff_description(
                    1, len(dataChannels))

            with self.dataSet.writer_for_analysis_images(
                    self, 'aligned_fiducial_images', fov) as outputTif:
                for t, x in zip(transformationList, dataChannels):
                    inputImage = self.dataSet.get_fiducial_image(x, fov)
                    transformedImage = transform.warp(
                            inputImage, t, preserve_range=True) \
                        .astype(inputImage.dtype)
                    outputTif.save(
                            transformedImage, 
                            photometric='MINISBLACK',
                            metadata=fiducialImageDescription)

        self._save_transformations(transformationList, fov)

    def _save_transformations(self, transformationList: List, fov: int) -> None:
        self.dataSet.save_numpy_analysis_result(
            np.array(transformationList), 'offsets',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations')

    def get_transformation(self, fov: int, dataChannel: int=None
                            ) -> Union[transform.EuclideanTransform,
                                 List[transform.EuclideanTransform]]:
        """Get the transformations for aligning images for the specified field
        of view.

        Args:
            fov: the fov to get the transformations for.
            dataChannel: the index of the data channel to get the transformation
                for. If None, then all data channels are returned.
        Returns:
            a EuclideanTransform if dataChannel is specified or a list of
                EuclideanTransforms for all dataChannels if dataChannel is
                not specified.
        """
        transformationMatrices = self.dataSet.load_numpy_analysis_result(
            'offsets', self, resultIndex=fov, subdirectory='transformations')
        if dataChannel is not None:
            return transformationMatrices[dataChannel]
        else:
            return transformationMatrices


class FiducialFitWarp(Warp):

    """
    An analysis task that warps a set of images taken in different imaging
    rounds based on fitting fiducial spots.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'initial_sigma' not in self.parameters:
            self.parameters['initial_sigma'] = 1.6
        if 'intensity_threshold' not in self.parameters:
            self.parameters['intensity_threshold'] = 10
        if 'significance_threshold' not in self.parameters:
            self.parameters['significance_threshold'] = 200

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return []

    def _run_analysis(self, fragmentIndex):
        self._fit_fiducials(fragmentIndex)
        localizations = self.load_fiducials(fragmentIndex)

        tforms = []
        referencePoints = self._extract_coordinates(
            localizations[0], self.parameters['significance_threshold'])
        for i, currentLocalizations in enumerate(localizations):
            movingPoints = self._extract_coordinates(currentLocalizations)
            rc, mc = registration.extract_control_points(
                referencePoints, movingPoints)
            tforms.append(registration.estimate_transform_from_points(rc, mc))
        
        self._process_transformations(tforms, fragmentIndex)

    def _fit_fiducials(self, fov: int) -> None:
        """Fit the fiducial spots for all channels for the specified fov.

        The output files containing the fiducial fits are saved into the data
        set directory for this analysis task.

        Args:
            fov: index of the field of view
        """
        for dataChannel in self.dataSet.get_data_organization()\
                .get_data_channels():
            fiducialFrame = self.dataSet.get_data_organization()\
                .get_fiducial_frame_index(dataChannel)
            fiducialName = self.dataSet.get_data_organization()\
                .get_fiducial_filename(dataChannel, fov)
            destPath = self.dataSet.get_analysis_subdirectory(
                    self.analysisName, subdirectory='fiducials')

            baseName = '_'.join(
                        [os.path.split(fiducialName)[1].split('.')[0],
                            str(dataChannel), str(fov)])
            parametersName = os.sep.join(
                    [destPath, baseName + '_parameters.xml'])
            outputName = os.sep.join([destPath, baseName + '_spots.hdf5'])

            params = self._generate_default_daostorm_parameters(
                fiducialFrame, fiducialFrame + 1)
            params.toXMLFile(parametersName)

            if not os.path.exists(outputName):
                mfit.analyze(fiducialName, outputName, parametersName)
            else:
                print(outputName + ' already exists')

    def _transform_fiducials_for_image_orientation(
            self, fiducialList: List[Dict]):
        """Transform the list of fiducials to correspond with image
        transformations specified by the data set

        The input list is modified and no copy is made.

        Args:
            fiducialList: list of fiducial information
        """
        if self.dataSet.transpose:
            for f in fiducialList:
                oldX = f['x']
                f['x'] = f['y']
                f['y'] = oldX
        
        if self.dataSet.flipHorizontal:
            for f in fiducialList:
                f['x'] = self.dataSet.imageDimensions[0] - np.array(f['x'])

        if self.dataSet.flipVertical:
            for f in fiducialList:
                f['y'] = self.dataSet.imageDimensions[1] - np.array(f['y'])

    def load_fiducials(self, fov: int) -> List[Dict]:
        """Load the fiducials fit to all fiducial frames for the specified fov.

        Before loading fiducials, they must first be fit using the
        _fit_fiducials function for the specified fov.

        Args:
            fov: index of the field of view
        Returns:
            A list of fiducial information where each index in the list
                corresponds to a data channel.
        """
        fiducials = []
        for i, dataChannel in enumerate(self.dataSet.get_data_organization()\
                .get_data_channels()):
            fiducialFrame = self.dataSet.get_data_organization()\
                .get_fiducial_frame_index(dataChannel)
            fiducialName = self.dataSet.get_data_organization()\
                .get_fiducial_filename(dataChannel, fov)
            destPath = self.dataSet.get_analysis_subdirectory(
                    self.analysisName, subdirectory='fiducials')

            baseName = '_'.join([os.path.split(fiducialName)[1].split('.')[0],
                                 str(dataChannel), str(fov)])
            outputName = os.sep.join([destPath, baseName + '_spots.hdf5'])

            fiducials.append(
                    saH5Py.SAH5Py(outputName).getLocalizationsInFrame(
                        int(fiducialFrame)))

        self._transform_fiducials_for_image_orientation(fiducials)

        return fiducials

    @staticmethod
    def _extract_coordinates(
            localizationSet: Dict,
            significanceThreshold: float=50) -> np.ndarray:
        return np.array([[x, y] for x, y, s
                         in zip(localizationSet['x'], localizationSet['y'],
                                localizationSet['significance'])
                         if s > significanceThreshold])

    def _generate_default_daostorm_parameters(self, startFrame: int=-1,
                                              endFrame: int=-1
                                              ) -> saparameters.ParametersDAO:
        params = saparameters.ParametersDAO()

        params.setAttr("max_frame", "int", endFrame)    
        params.setAttr("start_frame", "int", startFrame)
        
        params.setAttr("background_sigma", "float", 20.0)
        params.setAttr("camera_gain", "float", 1)
        params.setAttr("camera_offset", "float", 100)
        params.setAttr("find_max_radius", "int", 5)
        params.setAttr("foreground_sigma", "float",
                       self.parameters['initial_sigma'])
        params.setAttr("iterations", "int", 1)
        params.setAttr("model", "string", '2d')
        params.setAttr("pixel_size", "float", 106)
        params.setAttr("sigma", "float", self.parameters['initial_sigma'])
        params.setAttr("threshold", "float",
                       self.parameters['intensity_threshold'])

        params.setAttr("descriptor", "string", "1")
        params.setAttr("radius", "float", 0)

        params.setAttr("d_scale", "int", 2)
        params.setAttr("drift_correction", "int", 0)
        params.setAttr("frame_step", "int", 500)
        params.setAttr("z_correction", "int", 0)

        return params


class FiducialCorrelationWarp(Warp):

    """
    An analysis task that warps a set of images taken in different imaging
    rounds based on the crosscorrelation between fiducial images.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return []

    def _run_analysis(self, fragmentIndex: int):
        # TODO - this can be more efficient since some images should
        # use the same alignment if they are from the same imaging round
        fixedImage = self.dataSet.get_fiducial_image(0, fragmentIndex)
        offsets = [feature.register_translation(
            fixedImage, self.dataSet.get_fiducial_image(x, fragmentIndex), 100)[0]
               for x in self.dataSet.get_data_organization().get_data_channels()]
        transformations = [transform.SimilarityTransform(
            translation=[-x[1], -x[0]]) for x in offsets]
        self._process_transformations(transformations, fragmentIndex)
