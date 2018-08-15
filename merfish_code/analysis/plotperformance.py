
from merfish_code.core import analysistask

class PlotPerformance(analysistask.AnalysisTask):

    '''An analysis task that generates plots depicting metrics of the MERFISH
    decoding.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 30

    def run_analysis(self):
        #TODO - barcode intensity distrubiton plot
        #TODO - barcode pixel count distribution plot
        #TODO - barcode correlation plots
        #TODO - alignment error plots - need to save transformation information
        # first
        #TODO - good barcode and blank spatial distributions
        #TODO - barcode size spatial distribution
        #TODO - barcode distance spatial distribution
        #TODO - barcode intensity spatial distribution
        #TODO - abundance per barcode with blanks
        #TODO - confidence ratio per barcode with blanks
        #TODO - optimization convergence
        #TODO - good barcodes/blanks per cell
        pass
