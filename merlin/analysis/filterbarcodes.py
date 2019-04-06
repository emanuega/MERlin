from merlin.core import analysistask
from merlin.util import barcodedb


class FilterBarcodes(analysistask.ParallelAnalysisTask):

    """
    An analysis task that filters barcodes based on area and mean
    intensity.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'area_threshold' not in self.parameters:
            self.parameters['area_threshold'] = 3
        if 'intensity_threshold' not in self.parameters:
            self.parameters['intensity_threshold'] = 200

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_barcode_database(self):
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['decode_task']]

    def _run_analysis(self, fragmentIndex):
        decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])
        areaThreshold = self.parameters['area_threshold']
        intensityThreshold = self.parameters['intensity_threshold']
        barcodeDB = self.get_barcode_database()
        currentBC = decodeTask.get_barcode_database() \
                .get_filtered_barcodes(
                    areaThreshold,
                    intensityThreshold,
                    fov=fov)
        barcodeDB.write_barcodes(currentBC, fov=fov)
