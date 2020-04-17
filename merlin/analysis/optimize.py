import numpy as np
import itertools
from skimage import transform
from typing import Dict
from typing import List
import pandas

from merlin.analysis import decode
from merlin.util import decoding
from merlin.util import registration
from merlin.util import aberration
from merlin.data.codebook import Codebook


class OptimizeIteration(decode.BarcodeSavingParallelAnalysisTask):

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
        if 'optimize_background' not in self.parameters:
            self.parameters['optimize_background'] = False
        if 'optimize_chromatic_correction' not in self.parameters:
            self.parameters['optimize_chromatic_correction'] = False
        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 0

        if 'fov_index' in self.parameters:
            logger = self.dataSet.get_logger(self)
            logger.info('Setting fov_per_iteration to length of fov_index')

            self.parameters['fov_per_iteration'] = \
                len(self.parameters['fov_index'])

        else:
            self.parameters['fov_index'] = []
            for i in range(self.parameters['fov_per_iteration']):
                fovIndex = int(np.random.choice(
                    list(self.dataSet.get_fovs())))
                zIndex = int(np.random.choice(
                    list(range(len(self.dataSet.get_z_positions())))))
                self.parameters['fov_index'].append([fovIndex, zIndex])

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

    def get_codebook(self) -> Codebook:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        return preprocessTask.get_codebook()

    def _run_analysis(self, fragmentIndex):
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        codebook = self.get_codebook()

        fovIndex, zIndex = self.parameters['fov_index'][fragmentIndex]

        scaleFactors = self._get_previous_scale_factors()
        backgrounds = self._get_previous_backgrounds()
        chromaticTransformations = \
            self._get_previous_chromatic_transformations()

        self.dataSet.save_numpy_analysis_result(
            scaleFactors, 'previous_scale_factors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            backgrounds, 'previous_backgrounds', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_pickle_analysis_result(
            chromaticTransformations, 'previous_chromatic_corrections',
            self.analysisName, resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            np.array([fovIndex, zIndex]), 'select_frame', self.analysisName,
            resultIndex=fragmentIndex)

        chromaticCorrector = aberration.RigidChromaticCorrector(
            chromaticTransformations, self.get_reference_color())
        warpedImages = preprocessTask.get_processed_image_set(
            fovIndex, zIndex=zIndex, chromaticCorrector=chromaticCorrector)

        decoder = decoding.PixelBasedDecoder(codebook)
        areaThreshold = self.parameters['area_threshold']
        decoder.refactorAreaThreshold = areaThreshold
        di, pm, npt, d = decoder.decode_pixels(warpedImages, scaleFactors,
                                               backgrounds)

        refactors, backgrounds, barcodesSeen = \
            decoder.extract_refactors(
                di, pm, npt, extractBackgrounds=self.parameters[
                    'optimize_background'])

        # TODO this saves the barcodes under fragment instead of fov
        # the barcodedb should be made more general
        cropWidth = self.parameters['crop_width']
        self.get_barcode_database().write_barcodes(
            pandas.concat([decoder.extract_barcodes_with_index(
                i, di, pm, npt, d, fovIndex, cropWidth,
                zIndex, minimumArea=areaThreshold)
                for i in range(codebook.get_barcode_count())]),
            fov=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            refactors, 'scale_refactors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            backgrounds, 'background_refactors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            barcodesSeen, 'barcode_counts', self.analysisName,
            resultIndex=fragmentIndex)

    def _get_used_colors(self) -> List[str]:
        dataOrganization = self.dataSet.get_data_organization()
        codebook = self.get_codebook()
        return sorted({dataOrganization.get_data_channel_color(
            dataOrganization.get_data_channel_for_bit(x))
            for x in codebook.get_bit_names()})

    def _calculate_initial_scale_factors(self) -> np.ndarray:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        bitCount = self.get_codebook().get_bit_count()

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

    def _get_previous_backgrounds(self) -> np.ndarray:
        if 'previous_iteration' not in self.parameters:
            backgrounds = np.zeros(self.get_codebook().get_bit_count())
        else:
            previousIteration = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration'])
            backgrounds = previousIteration.get_backgrounds()

        return backgrounds

    def _get_previous_chromatic_transformations(self)\
            -> Dict[str, Dict[str, transform.SimilarityTransform]]:
        if 'previous_iteration' not in self.parameters:
            usedColors = self._get_used_colors()
            return {u: {v: transform.SimilarityTransform()
                        for v in usedColors if v >= u} for u in usedColors}
        else:
            previousIteration = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration'])
            return previousIteration._get_chromatic_transformations()

    # TODO the next two functions could be in a utility class. Make a
    #  chromatic aberration utility class

    def get_reference_color(self):
        return min(self._get_used_colors())

    def get_chromatic_corrector(self) -> aberration.ChromaticCorrector:
        """Get the chromatic corrector estimated from this optimization
        iteration

        Returns:
            The chromatic corrector.
        """
        return aberration.RigidChromaticCorrector(
            self._get_chromatic_transformations(), self.get_reference_color())

    def _get_chromatic_transformations(self) \
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

        if not self.parameters['optimize_chromatic_correction']:
            usedColors = self._get_used_colors()
            return {u: {v: transform.SimilarityTransform()
                        for v in usedColors if v >= u} for u in usedColors}

        try:
            return self.dataSet.load_pickle_analysis_result(
                'chromatic_corrections', self.analysisName)
        # OSError and ValueError are raised if the previous file is not
        # completely written
        except (FileNotFoundError, OSError, ValueError):
            # TODO - this is messy. It can be broken into smaller subunits and
            # most parts could be included in a chromatic aberration class
            previousTransformations = \
                self._get_previous_chromatic_transformations()
            previousCorrector = aberration.RigidChromaticCorrector(
                previousTransformations, self.get_reference_color())
            codebook = self.get_codebook()
            dataOrganization = self.dataSet.get_data_organization()

            barcodes = self.get_barcode_database().get_barcodes()
            uniqueFOVs = np.unique(barcodes['fov'])
            warpTask = self.dataSet.load_analysis_task(
                self.parameters['warp_task'])

            usedColors = self._get_used_colors()
            colorPairDisplacements = {u: {v: [] for v in usedColors if v >= u}
                                      for u in usedColors}

            for fov in uniqueFOVs:

                fovBarcodes = barcodes[barcodes['fov'] == fov]
                zIndexes = np.unique(fovBarcodes['z'])
                for z in zIndexes:
                    currentBarcodes = fovBarcodes[fovBarcodes['z'] == z]
                    # TODO this can be moved to the run function for the task
                    # so not as much repeated work is done when it is called
                    # from many different tasks in parallel
                    warpedImages = np.array([warpTask.get_aligned_image(
                        fov, dataOrganization.get_data_channel_for_bit(b),
                        int(z),  previousCorrector)
                        for b in codebook.get_bit_names()])

                    for i, cBC in currentBarcodes.iterrows():
                        onBits = np.where(
                            codebook.get_barcode(cBC['barcode_id']))[0]

                        # TODO this can be done by crop width when decoding
                        if cBC['x'] > 10 and cBC['y'] > 10 \
                                and warpedImages.shape[1]-cBC['x'] > 10 \
                                and warpedImages.shape[2]-cBC['y'] > 10:

                            refinedPositions = np.array(
                                [registration.refine_position(
                                    warpedImages[i, :, :], cBC['x'], cBC['y'])
                                    for i in onBits])
                            for p in itertools.combinations(
                                    enumerate(onBits), 2):
                                c1 = dataOrganization.get_data_channel_color(
                                    p[0][1])
                                c2 = dataOrganization.get_data_channel_color(
                                    p[1][1])

                                if c1 < c2:
                                    colorPairDisplacements[c1][c2].append(
                                        [np.array([cBC['x'], cBC['y']]),
                                         refinedPositions[p[1][0]]
                                         - refinedPositions[p[0][0]]])
                                else:
                                    colorPairDisplacements[c2][c1].append(
                                        [np.array([cBC['x'], cBC['y']]),
                                         refinedPositions[p[0][0]]
                                         - refinedPositions[p[1][0]]])

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
        # OSError and ValueError are raised if the previous file is not
        # completely written
        except (FileNotFoundError, OSError, ValueError):
            refactors = np.array([self.dataSet.load_numpy_analysis_result(
                    'scale_refactors', self.analysisName, resultIndex=i)
                for i in range(self.parameters['fov_per_iteration'])])

            # Don't rescale bits that were never seen
            refactors[refactors == 0] = 1

            previousFactors = np.array([self.dataSet.load_numpy_analysis_result(
                'previous_scale_factors', self.analysisName, resultIndex=i)
                for i in range(self.parameters['fov_per_iteration'])])

            scaleFactors = np.nanmedian(
                    np.multiply(refactors, previousFactors), axis=0)

            self.dataSet.save_numpy_analysis_result(
                scaleFactors, 'scale_factors', self.analysisName)

            return scaleFactors

    def get_backgrounds(self) -> np.ndarray:
        if not self.is_complete():
            raise Exception('Analysis is still running. Unable to get ' +
                            'backgrounds.')

        try:
            return self.dataSet.load_numpy_analysis_result(
                'backgrounds', self.analysisName)
        # OSError and ValueError are raised if the previous file is not
        # completely written
        except (FileNotFoundError, OSError, ValueError):
            refactors = np.array([self.dataSet.load_numpy_analysis_result(
                    'background_refactors', self.analysisName, resultIndex=i)
                for i in range(self.parameters['fov_per_iteration'])])

            previousBackgrounds = np.array(
                [self.dataSet.load_numpy_analysis_result(
                    'previous_backgrounds', self.analysisName, resultIndex=i)
                    for i in range(self.parameters['fov_per_iteration'])])

            previousFactors = np.array([self.dataSet.load_numpy_analysis_result(
                'previous_scale_factors', self.analysisName, resultIndex=i)
                for i in range(self.parameters['fov_per_iteration'])])

            backgrounds = np.nanmedian(np.add(
                previousBackgrounds, np.multiply(refactors, previousFactors)),
                axis=0)

            self.dataSet.save_numpy_analysis_result(
                backgrounds, 'backgrounds', self.analysisName)

            return backgrounds

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
            return np.append(
                previousHistory, [self.get_scale_factors()], axis=0)

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
            return np.append(previousHistory, [countsMean], axis=0)
