import os

import numpy as np

from sklearn.neighbors import NearestNeighbors
from skimage import transform
from skimage import feature

import storm_analysis.sa_library.parameters as parameters
import storm_analysis.daostorm_3d.mufit_analysis as mfit
import storm_analysis.sa_library.sa_h5py as saH5Py

from merlin.core import analysistask

class Warp(analysistask.ParallelAnalysisTask):

    '''
    An abstract class for warping a set of images so that the corresponding
    pixels align between images taken in different imaging rounds.
    '''

    def get_transformation(self, fov, dataChannel):
        transformations = self.dataSet.load_analysis_result(
                'offsets', self.get_analysis_name(), resultIndex=fov,
                subdirectory='transformations')
        return transformations[dataChannel]

    def _save_transformations(self, transformationList, fov):
        self.dataSet.save_analysis_result(
                np.array(transformationList), 'offsets',
                self.get_analysis_name(), resultIndex=fov,
                subdirectory='transformations')

    def get_aligned_image_set(self, fov):
        return self.dataSet.get_analysis_image_set(
                self, 'aligned_images', fov)

    def get_aligned_image(self, fov, dataChannel, zIndex):
        return self.dataSet.get_analysis_image(
                self, 'aligned_images', fov, 
                len(self.dataSet.get_z_positions()), dataChannel, zIndex)

    def _process_transformations(self, transformationList, fov):
        '''
        Process the transformations determined for a given fov. 

        The list of transformation is used to write registered images and 
        the transformation list is archived.

        Args:
            transformationList: A list of transformations that contains a
                transformation for each data channel. 
            fov: The fov that is being transformed.
        '''

        imageDescription = self.dataSet._analysis_tiff_description(
                len(self.dataSet.get_z_positions()),
                len(self.dataSet.get_data_channels()))

        with self.dataSet._writer_for_analysis_images(
                self, 'aligned_images', fov) as outputTif:
            for t,x in zip(
                    transformationList, self.dataSet.get_data_channels()):
                for z in self.dataSet.get_z_positions():
                    inputImage = self.dataSet.get_raw_image(x, fov, z)
                    transformedImage = transform.warp(
                            inputImage, t, preserve_range=True) \
                                    .astype(inputImage.dtype)
                    outputTif.save(
                            transformedImage, 
                            photometric='MINISBLACK',
                            metadata=imageDescription)

        self._save_transformations(transformationList, fov)


class FiducialFitWarp(Warp):

    '''
    An analysis task that warps a set of images taken in different imaging
    rounds based on fitting fiducials.
    '''


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

    def run_analysis(self, fragmentIndex):
        self.fit_fiducials(fragmentIndex)
        localizations = self.load_fiducials(fragmentIndex)

        tforms = []
        referencePoints = self.extract_coordinates(localizations[0])
        for currentLocalizations in localizations:
            movingPoints = self.extract_coordinates(currentLocalizations)
            rc, mc = self.extract_control_points(referencePoints, movingPoints)
            tforms.append(self.estimate_affine_transform(rc, mc))
        
        self._process_transformations(tforms, fragmentIndex)

    def fit_fiducials(self, fov):
        for dataChannel in self.dataSet.get_data_channels():
            fiducialFrame = self.dataSet.get_fiducial_frame(dataChannel)
            fiducialName = self.dataSet.get_fiducial_filename(dataChannel, fov)
            destPath = self.dataSet.get_analysis_subdirectory(
                    self.analysisName, subdirectory='fiducials')

            baseName = '_'.join(
                        [os.path.split(fiducialName)[1].split('.')[0], \
                            str(dataChannel), str(fov)])
            parametersName = os.sep.join(
                    [destPath, baseName + '_parameters.xml'])
            outputName = os.sep.join([destPath, baseName + '_spots.hdf5'])

            params = self.getDefaultParameters(fiducialFrame, fiducialFrame+1)
            params.toXMLFile(parametersName)

            if not os.path.exists(outputName):
                mfit.analyze(fiducialName, outputName, parametersName)
            else:
                print(outputName + ' already exists')

    def load_fiducials(self, fov):
        fiducials = []
        for dataChannel in self.dataSet.get_data_channels():
            fiducialFrame = self.dataSet.get_fiducial_frame(dataChannel)
            fiducialName = self.dataSet.get_fiducial_filename(dataChannel, fov)
            destPath = self.dataSet.get_analysis_subdirectory(
                    self.analysisName, subdirectory='fiducials')

            baseName = '_'.join([os.path.split(fiducialName)[1].split('.')[0], \
                            str(dataChannel), str(fov)])
            parametersName = os.sep.join(
                    [destPath, baseName + '_parameters.xml'])
            outputName = os.sep.join([destPath, baseName + '_spots.hdf5'])

            fiducials.append(
                    saH5Py.SAH5Py(outputName).getLocalizationsInFrame(
                        int(fiducialFrame)))

        return fiducials

    def extract_coordinates(self, localizationSet):
        return np.array(
            [[x, y] for x,y in zip(localizationSet['x'], localizationSet['y'])])

    def extract_control_points(self, referencePoints, movingPoints):
        edgeSpacing = 0.5
        edges = np.arange(-30, 30, edgeSpacing)

        neighbors = NearestNeighbors(n_neighbors=10)
        neighbors.fit(referencePoints)
        distances, indexes = neighbors.kneighbors(
                movingPoints, return_distance=True)
        differences = [[referencePoints[x] - movingPoints[i] \
                for x in indexes[i]] \
                for i in range(len(movingPoints))]
        counts, xedges, yedges = np.histogram2d(
                [x[0] for y in differences for x in y],
                [x[1] for y in differences for x in y],
                bins = edges)
        maxIndex = np.unravel_index(counts.argmax(), counts.shape)
        offset = (-xedges[maxIndex[0]], -yedges[maxIndex[1]])

        distancesShifted, indexesShifted = neighbors.kneighbors(
                movingPoints-np.tile(offset, (movingPoints.shape[0], 1)),
                return_distance=True)

        controlIndexes = [x[0] < edgeSpacing for x in distancesShifted]
        referenceControls = np.array([referencePoints[x[0]] \
                for x in indexesShifted[controlIndexes]])
        movingControls = movingPoints[controlIndexes,:]

        return referenceControls, movingControls

    def estimate_affine_transform(self, referenceControls, movingControls):
        tform = transform.AffineTransform()
        tform.estimate(referenceControls, movingControls)
        return tform

    def getDefaultParameters(self, startFrame = -1, endFrame = -1):
        params = parameters.ParametersDAO()

        params.setAttr("max_frame", "int", endFrame)    
        params.setAttr("start_frame", "int", startFrame)
        
        params.setAttr("background_sigma", "float", 20.0)
        params.setAttr("camera_gain", "float", 1)
        params.setAttr("camera_offset", "float", 100)
        params.setAttr("find_max_radius", "int", 5)
        params.setAttr("foreground_sigma", "float", 2.0)
        params.setAttr("iterations", "int", 1)
        params.setAttr("model", "string", '2dfixed')
        params.setAttr("pixel_size", "float", 106)
        params.setAttr("roi_size", "int", 9)
        params.setAttr("sigma", "float", 2)
        params.setAttr("threshold", "float", 6.0)

        # Do tracking.
        params.setAttr("descriptor", "string", "1")
        params.setAttr("radius", "float", 0)

        # Do drift-correction.
        params.setAttr("d_scale", "int", 2)
        params.setAttr("drift_correction", "int", 0)
        params.setAttr("frame_step", "int", 500)
        params.setAttr("z_correction", "int", 0)

        return params

class FiducialCorrelationWarp(Warp):

    '''
    An analysis task that warps a set of images taken in different imaging
    rounds based on the crosscorrelation between fiducial images.
    '''

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

    def run_analysis(self, fragmentIndex):
        #TODO - this can be more efficient since some images should
        #use the same aligned if they are from the same imaging round
        fixedImage = self.dataSet.get_fiducial_image(0, fragmentIndex)
        offsets = [feature.register_translation(
                        fixedImage, 
                        self.dataSet.get_fiducial_image(x, fragmentIndex),
                        100)[0] \
                    for x in self.dataSet.get_data_channels()]
        transformations = [transform.SimilarityTransform(
            translation=[-x[1], -x[0]]) for x in offsets]
        self._process_transformations(transformations, fragmentIndex)
