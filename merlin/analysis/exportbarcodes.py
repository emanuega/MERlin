from merlin.core import analysistask


class ExportBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that filters barcodes based on area and mean 
    intensity.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'columns' not in self.parameters:
            self.parameters['columns'] = ['barcode_id', 'global_x',
                                          'global_y', 'cell_index']
        if 'exclude_blanks' not in self.parameters:
            self.parameters['exclude_blanks'] = True

        self.columns = self.parameters['columns']
        self.excludeBlanks = self.parameters['exclude_blanks']

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['filter_task']]

    def run_analysis(self):
        filterTask = self.dataSet.load_analysis_task(
                self.parameters['filter_task'])        

        barcodeData = filterTask.get_barcode_database() \
                .get_barcodes(columnList=self.columns)

        if self.excludeBlanks:
            codebook = self.dataSet.get_codebook()
            barcodeData = barcodeData[
                    barcodeData['barcode_id'].isin(
                        codebook.get_coding_indexes())]

        self.dataSet.save_dataframe_to_csv(barcodeData, 'barcodes', self)
