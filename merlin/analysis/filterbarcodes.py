from merlin.core import analysistask
from merlin.util import barcodedb


class FilterBarcodes(analysistask.AnalysisTask):

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

        self.areaThreshold = self.parameters['area_threshold']
        self.intensityThreshold = self.parameters['intensity_threshold']

    def get_barcode_database(self):
        return barcodedb.SQLiteBarcodeDB(self.dataSet, self)

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['decode_task']]

    def _run_analysis(self):
        decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])        

        barcodeDB = self.get_barcode_database()
        for fov in self.dataSet.get_fovs():
            for currentBC in decodeTask.get_barcode_database() \
                    .get_filtered_barcodes(
                        self.areaThreshold, self.intensityThreshold, 
                        fov=fov, chunksize=10000):
                barcodeDB.write_barcodes(currentBC, fov=fov)
