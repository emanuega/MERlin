import pandas
import rtree
import networkx
import numpy as np
import cv2
from skimage.measure import regionprops

from merlin.core import analysistask
from merlin.util import filter


class SumSignal(analysistask.ParallelAnalysisTask):

    """
    An analysis task that calculates the signal intensity within the boundaries
    of a cell for all rounds not used in the codebook, useful for measuring
    RNA species that were stained individually.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'apply_highpass' not in self.parameters:
            self.parameters['apply_highpass'] = True
        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 5

        self.highpass = str(self.parameters['apply_highpass']).upper() == 'TRUE'
        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def _extract_signal(self, cells, inputImage, zIndex) -> pandas.DataFrame:
        cellCoords = []
        for cell in cells:
            regions = cell.get_boundaries()[zIndex]
            if len(regions) == 0:
                cellCoords.append([])
            else:
                pixels = []
                for region in regions:
                    coords = region.exterior.coords.xy
                    xyZip = list(zip(coords[0].tolist(), coords[1].tolist()))
                    pixels.append(np.array(
                                self.alignTask.global_coordinates_to_fov(
                                    cell.get_fov(), xyZip)))
                cellCoords.append(pixels)
        # keptCells and keptCellIDs prevent cells with no area from getting
        # through, isn't strictly necessary if get_intersection_graph
        # is run with an area threshold
        keptCells = [cellCoords[x] for x in range(len(cells))
                     if len(cellCoords[x]) > 0]
        keptCellIDs = [str(cells[x].get_feature_id()) for x in range(len(cells))
                       if len(cellCoords[x]) > 0]
        mask = np.zeros(inputImage.shape, np.uint8)
        for i, cell in enumerate(keptCells):
            cv2.drawContours(mask, cell, -1, i+1, -1)
        props = regionprops(mask, inputImage)
        propsOut = pandas.DataFrame(
            data=[(x.intensity_image.sum(), x.filled_area) for x in props],
            index=keptCellIDs,
            columns=['Intensity', 'Pixels'])
        return propsOut

    @staticmethod
    def get_intersection_graph(polygonList, areaThreshold=250):
        # This is only currently necessary to eliminate
        # problematic cell overlaps. If cell boundaries have been cleaned
        # prior to running sum signal this isn't necessary
        polygonIndex = rtree.index.Index()
        intersectGraphEdges = [[i, i] for i in range(len(polygonList))]
        for i, cell in enumerate(polygonList):
            if len(cell.get_bounding_box()) == 4:
                putativeIntersects = list(polygonIndex.intersection(
                                          cell.get_bounding_box()))
                if len(putativeIntersects) > 0:
                    try:
                        intersectGraphEdges += \
                                [[i, j] for j in putativeIntersects
                                 if cell.intersection(
                                    polygonList[j]) > areaThreshold]
                    except Exception:
                        print('Unable to calculate intersection for cell %i'
                              % i)

                polygonIndex.insert(i, cell.get_bounding_box())

        intersectionGraph = networkx.Graph()
        intersectionGraph.add_edges_from(intersectGraphEdges)

        return intersectionGraph

    def _get_sum_signal(self, fov, channels, zIndex):

        fTask = self.dataSet.load_analysis_task(self.parameters['warp_task'])
        sTask = self.dataSet.load_analysis_task(self.parameters['segment_task'])

        cells = sTask.get_feature_database().read_features(fov)

        # If cell boundaries are going to be cleaned prior to this we should
        # remove the part enclosed by pound symbols
        ig = self.get_intersection_graph(cells, areaThreshold=0)

        cellIndex = [x for x in ig.nodes() if len(ig.edges(nbunch=x)) == 1
                     and cells[x].get_volume() > 0.0]
        cells = [cells[x] for x in range(len(cells)) if x in cellIndex]

        signals = []
        for ch in channels:
            img = fTask.get_aligned_image(fov, ch, zIndex)
            if self.highpass:
                img = filter.high_pass_filter(
                    img, self.parameters['highpass_sigma'])
            signals.append(self._extract_signal(cells, img,
                                                zIndex).iloc[:, [0]])

        # adding num of pixels
        signals.append(self._extract_signal(cells, img, zIndex).iloc[:, [1]])

        compiledSignal = pandas.concat(signals, 1)
        compiledSignal.columns = channels+['Pixels']

        return compiledSignal

    def get_sum_signals(self, fov: int = None) -> pandas.DataFrame:
        """Retrieve the sum signals calculated from this analysis task.

        Args:
            fov: the fov to get the sum signals for. If not specified, the
                sum signals for all fovs are returned.

        Returns:
            A pandas data frame containing the sum signal information.
        """
        if fov is None:
            return pandas.concat(
                [self.get_sum_signals(fov) for fov in self.dataSet.get_fovs()]
            ).reset_index(drop=True)

        return self.dataSet.load_dataframe_from_csv(
            'sequential_signal', self.get_analysis_name(),
            fov, 'signals', index_col=0)

    def _run_analysis(self, fragmentIndex):
        zIndex = int(self.parameters['z_index'])
        channels, geneNames = self.dataSet.get_data_organization()\
            .get_sequential_rounds()

        fovSignal = self._get_sum_signal(fragmentIndex, channels, zIndex)
        normSignal = fovSignal.iloc[:, :-1].div(fovSignal.loc[:, 'Pixels'], 0)

        normSignal.columns = geneNames
        normSignal.reset_index(inplace=True)
        columns = normSignal.columns.values.tolist()
        columns[0] = 'cell ID'
        normSignal.columns = columns

        self.dataSet.save_dataframe_to_csv(
                normSignal, 'sequential_signal', self.get_analysis_name(),
                fragmentIndex, 'signals')


class ExportSumSignals(analysistask.AnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['sequential_task']]

    def _run_analysis(self):
        sTask = self.dataSet.load_analysis_task(
                    self.parameters['sequential_task'])
        signals = sTask.get_sum_signals()

        self.dataSet.save_dataframe_to_csv(
                    signals, 'sequential_signal_compiled',
                    self.get_analysis_name())
