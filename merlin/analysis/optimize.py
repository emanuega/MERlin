import random
import numpy as np
import multiprocessing
import itertools
from skimage import transform
from typing import Dict
from typing import List

from merlin.core import analysistask
from merlin.util import decoding
from merlin.util import barcodedb
from merlin.util import registration


class OptimizeIteration(analysistask.ParallelAnalysisTask):

    """
    An analysis task for performing a single iteration of scale factor
    optimization.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'fov_per_iteration' not in self.parameters:
            self.parameters['fov_per_iteration'] = 50
        if 'area_threshold' not in self.parameters:
            self.parameters['area_threshold'] = 5

    def get_estimated_memory(self):
        return 4000

    def get_estimated_time(self):
        return 60

    def get_dependencies(self):
        dependencies = [self.parameters['preprocess_task'],
                        self.parameters['warp_task']]
        if 'previous_iteration' in self.parameters:
            dependencies += [self.parameters['previous_iteration']]
        return dependencies

    def fragment_count(self):
        return self.parameters['fov_per_iteration']

    def get_barcode_database(self) -> barcodedb.BarcodeDB:
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)

    def _run_analysis(self, fragmentIndex):
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])

        codebook = self.dataSet.get_codebook()

        fovIndex = np.random.choice(list(self.dataSet.get_fovs()))
        zIndex = np.random.choice(
            list(range(len(self.dataSet.get_z_positions()))))

        usedColors = self._get_used_colors()
        referenceColor = min(usedColors)

        scaleFactors = self._get_previous_scale_factors()
        chromaticTransformations = \
            self._get_previous_chromatic_transformations()

        self.dataSet.save_numpy_analysis_result(
            scaleFactors, 'previous_scale_factors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_pickle_analysis_result(
            chromaticTransformations, 'previous_chromatic_corrections',
            self.analysisName, resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            np.array([fovIndex, zIndex]), 'select_frame', self.analysisName,
            resultIndex=fragmentIndex)

        imageSet = preprocessTask.get_processed_image_set(
            fovIndex, zIndex=zIndex)
        warpedImages = np.array([self._warp_image(
            image, i, chromaticTransformations, referenceColor)
            for i, image in enumerate(imageSet)])

        decoder = decoding.PixelBasedDecoder(codebook)
        areaThreshold = self.parameters['area_threshold']
        decoder.refactorAreaThreshold = areaThreshold
        di, pm, npt, d = decoder.decode_pixels(warpedImages, scaleFactors)

        refactors, barcodesSeen = decoder.extract_refactors(di, pm, npt)

        for i in range(codebook.get_barcode_count()):
            # TODO this saves the barcodes under fragment instead of fov
            # the barcodedb should be made more general
            self.get_barcode_database().write_barcodes(
                    decoder.extract_barcodes_with_index(
                        i, di, pm, npt, d, fovIndex,
                        self.dataSet.z_index_to_position(zIndex),
                        0, minimumArea=areaThreshold
                    ), fov=fragmentIndex)

        self.dataSet.save_numpy_analysis_result(
            refactors, 'refactors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            barcodesSeen, 'barcode_counts', self.analysisName,
            resultIndex=fragmentIndex)

    def _get_used_colors(self) -> List[str]:
        dataOrganization = self.dataSet.get_data_organization()
        codebook = self.dataSet.get_codebook()
        return sorted({dataOrganization.get_data_channel_color(
            dataOrganization.get_data_channel_index(x))
            for x in codebook.get_bit_names()})

    def _calculate_initial_scale_factors(self) -> np.ndarray:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])

        bitCount = self.dataSet.get_codebook().get_bit_count()
        initialScaleFactors = np.zeros(bitCount)
        pixelHistograms = preprocessTask.get_pixel_histogram()
        for i in range(bitCount):
            cumulativeHistogram = np.cumsum(pixelHistograms[i])
            cumulativeHistogram = cumulativeHistogram/cumulativeHistogram[-1]
            # Add two to match matlab code.
            # TODO: Does +2 make sense? Used to be consistent with Matlab code
            initialScaleFactors[i] = \
                np.argmin(np.abs(cumulativeHistogram-0.9)) + 2

        return initialScaleFactors

    def _get_previous_scale_factors(self) -> np.ndarray:
        if 'previous_iteration' not in self.parameters:
            scaleFactors = self._calculate_initial_scale_factors()
        else:
            previousIteration = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration'])
            scaleFactors = previousIteration.get_scale_factors()

        return scaleFactors

    def _get_previous_chromatic_transformations(self)\
            -> Dict[str, Dict[str, transform.SimilarityTransform]]:
        if 'previous_iteration' not in self.parameters:
            usedColors = self._get_used_colors()
            chromaticTransformations = {c: transform.SimilarityTransform()
                                        for c in usedColors}
        else:
            previousIteration = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration'])
            chromaticTransformations = \
                previousIteration.get_chromatic_transformations()
        return chromaticTransformations

    def _warp_image(self, imageIn, index, tset, referenceColor):
        imageColor = self.dataSet.data.get_data_organization().\
                         get_data_channel_color(index)
        if imageColor == referenceColor:
            return imageIn
        tForm = tset[referenceColor][imageColor]
        return transform.warp(imageIn, tForm, preserve_range=True)

    def get_chromatic_corrections(self) \
            -> Dict[str, Dict[str, transform.SimilarityTransform]]:
        """Get the estimated chromatic corrections from this optimization
        iteration.

        Returns:
            a dictionary of dictionary of transformations for transforming
            the farther red colors to the most blue color. The transformation
            for transforming the farther red color, e.g. '750', to the
            farther blue color, e.g. '560', is found at result['560']['750']
        """
        if not self.is_complete():
            raise Exception('Analysis is still running. Unable to get scale '
                            + 'factors.')

        try:
            return self.dataSet.load_pickle_analysis_result(
                'chromatic_corrections', self.analysisName)
        except FileNotFoundError:
            # TODO - this is messy. It can be broken into smaller subunits.
            previousTransformations = \
                self._get_previous_chromatic_transformations()
            codebook = self.dataSet.get_codebook()
            dataOrganization = self.dataSet.get_data_organization()

            barcodes = self.get_barcode_database().get_barcodes()
            uniqueFOVs = np.unique(barcodes['fov'])[0]
            warpTask = self.dataSet.load_analysis_task(
                self.parameters['warp_task'])

            usedColors = self._get_used_colors()
            colorPairDisplacements = {u: {v: [] for v in usedColors if v >= u}
                                      for u in usedColors}
            referenceColor = min(usedColors)

            for fov in uniqueFOVs:
                fovBarcodes = barcodes[barcodes['fov'] == fov]
                zIndexes = np.unique(fovBarcodes['z'])
                images = warpTask.get_aligned_image_set(fov=fov)
                for z in zIndexes:
                    currentBarcodes = fovBarcodes[fovBarcodes['z'] == z]
                    warpedImages = np.array(
                        [self._warp_image(image[i, z, 0, :, :], i,
                                          previousTransformations,
                                          referenceColor)
                         for i, image in enumerate(images)]
                    )
                    for i, cBC in currentBarcodes.iterrows():
                        onBits = np.where(
                            codebook.get_barcode(cBC['barcode_id']))[0]
                        refinedPositions = np.array(
                            [registration.refine_position(
                                warpedImages[i, :, :], cBC['x'], cBC['y'])
                                for i in onBits])
                        for p in itertools.combinations(onBits, 2):
                            c1 = str(dataOrganization.get_data_channel_color(
                                onBits[p[0]]))
                            c2 = str(dataOrganization.get_data_channel_color(
                                onBits[p[1]]))

                            if c1 < c2:
                                colorPairDisplacements[c1][c2].append(
                                    [np.array([cBC['x'], cBC['y']]),
                                     refinedPositions[p[1]] - refinedPositions[
                                         p[0]]])
                            else:
                                colorPairDisplacements[c2][c1].append(
                                    [np.array([cBC['x'], cBC['y']]),
                                     refinedPositions[p[0]] - refinedPositions[
                                         p[1]]])

            tForms = {}
            for k, v in colorPairDisplacements.items():
                tForms[k] = {}
                for k2, v2 in v.items():
                    tForm = transform.SimilarityTransform()
                    goodIndexes = [i for i, x in enumerate(v2) if
                                   not any(np.isnan(x[1])) and not any(
                                       np.isinf(x[1]))]
                    tForm.estimate(
                        np.array([v2[i][0] for i in goodIndexes]),
                        np.array([v2[i][0] + v2[i][1] for i in goodIndexes]))
                    tForms[k][k2] = tForm + previousTransformations[k][k2]

            self.dataSet.save_pickle_analysis_result(
                tForms, 'chromatic_corrections', self.analysisName)

            return tForms

    def get_scale_factors(self) -> np.ndarray:
        """Get the final, optimized scale factors.

        Returns:
            a one-dimensional numpy array where the i'th entry is the
            scale factor corresponding to the i'th bit.
        """
        if not self.is_complete():
            raise Exception('Analysis is still running. Unable to get scale '
                            + 'factors.')

        try:
            return self.dataSet.load_numpy_analysis_result(
                'scale_factors', self.analysisName)
        except FileNotFoundError:
            refactors = np.array([self.dataSet.load_numpy_analysis_result(
                    'refactors', self.analysisName, resultIndex=i)
                for i in range(self.parameters['fov_per_iteration'])])

            # Don't rescale bits that were never seen
            refactors[refactors==0] = 1

            previousFactors = np.array([self.dataSet.load_numpy_analysis_result(
                'previous_scale_factors', self.analysisName, resultIndex=i)
                for i in range(self.parameters['fov_per_iteration'])])

            scaleFactors = np.nanmedian(
                    np.multiply(refactors, previousFactors), axis=0)

            self.dataSet.save_numpy_analysis_result(
                scaleFactors, 'scale_factors', self.analysisName)

            return scaleFactors

    def get_scale_factor_history(self) -> np.ndarray:
        """Get the scale factors cached for each iteration of the optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the
            scale factor corresponding to the i'th bit in the j'th
            iteration.
        """
        if 'previous_iteration' not in self.parameters:
            return np.array([self.get_scale_factors()])
        else:
            previousHistory = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration']
            ).get_scale_factor_history()
            return np.append(previousHistory, [self.get_scale_factors()],
                    axis=0)

    def get_barcode_count_history(self) -> np.ndarray:
        """Get the set of barcode counts for each iteration of the
        optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the
            barcode count corresponding to the i'th barcode in the j'th
            iteration.
        """
        countsMean = np.mean([self.dataSet.load_numpy_analysis_result(
            'barcode_counts', self.analysisName, resultIndex=i)
            for i in range(self.parameters['fov_per_iteration'])], axis=0)

        if 'previous_iteration' not in self.parameters:
            return np.array([countsMean])
        else:
            previousHistory = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration']
            ).get_barcode_count_history()
            return np.append(previousHistory, [countsMean],
                    axis=0)


class Optimize(analysistask.InternallyParallelAnalysisTask):

    """
    An analysis task for optimizing the parameters used for assigning barcodes
    to the image data.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'iteration_count' not in self.parameters:
            self.parameters['iteration_count'] = 20
        if 'fov_per_iteration' not in self.parameters:
            self.parameters['fov_per_iteration'] = 10
        if 'estimate_initial_scale_factors_from_cdf' not in self.parameters:
            self.parameters['estimate_initial_scale_factors_from_cdf'] = False
        if 'area_threshold' not in self.parameters:
            self.parameters['area_threshold'] = 4

    def get_estimated_memory(self):
        return 4000*self.coreCount

    def get_estimated_time(self):
        return 60 

    def get_dependencies(self):
        return [self.parameters['preprocess_task']]

    def _run_analysis(self):
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        iterationCount = self.parameters['iteration_count']
        fovPerIteration = self.parameters['fov_per_iteration']

        codebook = self.dataSet.get_codebook()
        bitCount = codebook.get_bit_count()
        barcodeCount = codebook.get_barcode_count()
        decoder = decoding.PixelBasedDecoder(codebook)
        decoder.refactorAreaThreshold = self.parameters['area_threshold']

        scaleFactors = np.ones((iterationCount, bitCount))
        if self.parameters['estimate_initial_scale_factors_from_cdf']:
            scaleFactors[0, :] = self._calculate_initial_scale_factors()

        barcodeCounts = np.ones((iterationCount, barcodeCount))
        pool = multiprocessing.Pool(processes=self.coreCount)
        for i in range(1, iterationCount):
            fovIndexes = random.sample(
                    list(self.dataSet.get_fovs()), fovPerIteration)
            zIndexes = np.random.choice(
                    list(range(len(self.dataSet.get_z_positions()))),
                    fovPerIteration)
            decoder._scaleFactors = scaleFactors[i - 1, :]
            r = pool.starmap(decoder.extract_refactors,
                             ([preprocessTask.get_processed_image_set(
                                 f, zIndex=z)]
                                 for f, z in zip(fovIndexes, zIndexes)))
            scaleFactors[i, :] = scaleFactors[i-1, :] \
                                 * np.mean([x[0] for x in r], axis=0)
            barcodeCounts[i, :] = np.mean([x[1] for x in r], axis=0)

        self.dataSet.save_numpy_analysis_result(scaleFactors, 'scale_factors',
                                                self.analysisName)
        self.dataSet.save_numpy_analysis_result(barcodeCounts, 'barcode_counts',
                                                self.analysisName)

    def _calculate_initial_scale_factors(self) -> np.ndarray:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])

        bitCount = self.dataSet.get_codebook().get_bit_count()
        initialScaleFactors = np.zeros(bitCount)
        pixelHistograms = preprocessTask.get_pixel_histogram()
        for i in range(bitCount):
            cumulativeHistogram = np.cumsum(pixelHistograms[i])
            cumulativeHistogram = cumulativeHistogram/cumulativeHistogram[-1]
            # Add two to match matlab code.
            # TODO: Does +2 make sense?
            initialScaleFactors[i] = \
                np.argmin(np.abs(cumulativeHistogram-0.9)) + 2

        return initialScaleFactors

    def get_scale_factors(self) -> np.ndarray:
        """Get the final, optimized scale factors.

        Returns:
            a one-dimensional numpy array where the i'th entry is the 
            scale factor corresponding to the i'th bit.
        """
        return self.dataSet.load_numpy_analysis_result(
            'scale_factors', self.analysisName)[-1, :]

    def get_scale_factor_history(self) -> np.ndarray:
        """Get the scale factors cached for each iteration of the optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the 
            scale factor corresponding to the i'th bit in the j'th 
            iteration.
        """
        return self.dataSet.load_numpy_analysis_result('scale_factors',
                                                       self.analysisName)

    def get_barcode_count_history(self) -> np.ndarray:
        """Get the set of barcode counts for each iteration of the
        optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the
            barcode count corresponding to the i'th barcode in the j'th
            iteration.
        """
        return self.dataSet.load_numpy_analysis_result('barcode_counts',
                                                       self.analysisName)
