import os
import errno
from merlin.core import analysistask
from shutil import copy2

class BypassAnalyzedData(analysistask.AnalysisTask):
    """
    A metaanalysis task that imports your desired data file to use with subsequent methods.
    Currently designed for csv files
    """

    def __init__(self, metaDataSet, parameters=None, analysisName=None):
        super().__init__(metaDataSet, parameters, analysisName)

        if 'overwrite' not in self.parameters:
            self.parameters['overwrite'] = False

        self.metaDataSet = metaDataSet

    #If additional loading methods are added then update this list to reflect supported file types
    supported_ext = ['.csv']

    if not os.path.isfile(self.parameters['source_file']):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.parameters['source_file'])

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 100

    def get_dependencies(self):
        return []

    def _run_analysis(self):
        ext = os.path.splitext(self.parameters['source_file'])[1]
        if ext not in supported_ext:
            print('Currently only {} file types are supported with this analysis task'.format(', '.join(supported_ext)))
        else:
            dst = self.metaDataSet._analysis_result_save_path('starting_data', self, fileExtension=ext)
            copy2(self.parameters['source_file'], dst)

    def return_exported_data(self):
        ext = os.path.splitext(self.parameters['source_file'])[1]
        if ext == '.csv':
            return self.metaDataSet.load_dataframe_from_csv('starting_data', self)
        else:
            print('No method to load {} is currently supported'.format(ext))
